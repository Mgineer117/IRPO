import os
from copy import deepcopy

import torch
import torch.nn as nn

from policy.hrl import HRL_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo import PPO_Learner
from policy.uniform_random import UniformRandom
from trainer.hrl_trainer import HRLTrainer
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.sampler import HLSampler, OnlineSampler


class HRL(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(HRL, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        if self.args.intrinsic_reward_mode in ("eigenpurpose", "allo", "random"):
            self.intrinsic_reward_fn = IntrinsicRewardFunctions(
                logger=logger,
                writer=writer,
                args=args,
            )
        else:
            raise ValueError(
                f"The intrinsic reward mode { self.args.intrinsic_reward_mode} not suitable for HRL."
            )

        self.args.nupdates = args.timesteps // (
            args.minibatch_size * args.num_minibatch
        )
        self.args.hl_nupdates = args.hl_timesteps // (
            args.minibatch_size * args.num_minibatch
        )

        self.current_timesteps = 0

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        hl_sampler = HLSampler(
            state_dim=self.args.state_dim,
            action_dim=int(self.args.num_options + 1),
            episode_len=self.env.max_steps,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
            max_option_len=10,
            gamma=self.args.gamma,
            verbose=False,
        )

        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
            verbose=False,
        )

        trainer = HRLTrainer(
            env=self.env,
            hl_policy=self.hl_policy,
            policies=self.policies,
            intrinsic_reward_fn=self.intrinsic_reward_fn,
            hl_sampler=hl_sampler,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=self.args.timesteps,
            hl_timesteps=self.args.hl_timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
        )

        # design hl_policy and hl_sampler and trainer

        trainer.train()

    def define_policy(self):
        # === Define policy === #
        self.policies = nn.ModuleList([])
        for i in range(self.args.num_options):
            actor = PPO_Actor(
                input_dim=self.args.state_dim,
                hidden_dim=self.args.actor_fc_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
            )
            critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

            policy = PPO_Learner(
                actor=actor,
                critic=critic,
                is_discrete=self.args.is_discrete,
                nupdates=self.args.nupdates,
                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                num_minibatch=self.args.num_minibatch,
                minibatch_size=self.args.minibatch_size,
                eps_clip=self.args.eps_clip,
                entropy_scaler=self.args.entropy_scaler,
                target_kl=self.args.target_kl,
                gamma=self.args.gamma,  # 1.0,  # gamma for option is 1 to find maxima
                gae=self.args.gae,
                K=self.args.K_epochs,
                device=self.args.device,
            )
            policy.name = "HRL_options"
            self.policies.append(policy)

        uniform_random_policy = UniformRandom(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )

        self.policies.append(uniform_random_policy)

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=int(self.args.num_options + 1),
            is_discrete=True,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.hl_policy = HRL_Learner(
            actor=actor,
            critic=critic,
            is_discrete=self.args.is_discrete,
            nupdates=self.args.hl_nupdates,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            for p in self.policies:
                p.grid = self.env.get_grid()
            self.hl_policy.grid = self.env.get_grid()
