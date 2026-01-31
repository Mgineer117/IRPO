import torch.nn as nn

from policy.irpo import IRPO_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from trainer.irpo_trainer import IRPOTrainer
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.sampler import OnlineSampler


class IRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IRPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.intrinsic_reward_fn = IntrinsicRewardFunctions(
            logger=logger,
            writer=writer,
            args=args,
        )

        self.current_timesteps = self.intrinsic_reward_fn.current_timesteps

    def begin_training(self):
        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size,
        )

        # === Meta-train using options === #'
        self.define_outer_policy()
        trainer = IRPOTrainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
            args=self.args,
        )
        final_steps = trainer.train()
        self.current_timesteps += final_steps

    def define_outer_policy(self):
        # === Define policy === #
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.policy = IRPO_Learner(
            actor=actor,
            critic=critic,
            intrinsic_reward_fn=self.intrinsic_reward_fn,
            timesteps=self.args.timesteps,
            num_inner_updates=self.args.num_inner_updates,
            outer_level_update_mode=self.args.outer_level_update_mode,
            outer_actor_lr=self.args.outer_actor_lr,
            inner_actor_lr=self.args.inner_actor_lr,
            critic_lr=self.args.critic_lr,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()
