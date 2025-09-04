import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.rl import (
    compute_kl,
    conjugate_gradients,
    estimate_advantages,
    flat_params,
    hessian_vector_product,
    set_flat_params,
)
from utils.sampler import OnlineSampler


class IRPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        intrinsic_reward_fn: IntrinsicRewardFunctions,
        num_inner_updates: int,
        outer_level_update_mode: str = "trpo",
        outer_actor_lr: float = 3e-4,
        inner_actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.9,
        eps_clip: float = 0.2,
        weight_option: str = "softmax",
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # define name for future logging
        self.name = "IRPO"
        self.device = device

        # storing parameters
        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.is_discrete = is_discrete

        # IRPO parameters
        self.outer_level_update_mode = outer_level_update_mode  # "trpo" or "sgd"
        self.num_inner_updates = num_inner_updates

        # only applicable when outer_level_update_mode is "sgd"
        self.outer_actor_lr = outer_actor_lr
        # lr to derive the exploratiry policy
        self.inner_actor_lr = inner_actor_lr
        self.critic_lr = critic_lr

        # define algorithmic parameters
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        # outer-level KL constraint
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        # critic l2 regularization
        self.l2_reg = l2_reg

        self.irm_weight = 1.0

        # define neural networks
        self.actor = actor
        self.critic = critic
        # intrinsic reward function class
        self.intrinsic_reward_fn = intrinsic_reward_fn

        self.base_critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )

        # define critics for extrinsic and intrinsic rewards
        self.extrinsic_critics = nn.ModuleList(
            [deepcopy(self.critic) for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )
        self.intrinsic_critics = nn.ModuleList(
            [deepcopy(self.critic) for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )

        # define optimizers for extrinsic and intrinsic critics
        self.extrinsic_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.extrinsic_critics
        ]
        self.intrinsic_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.intrinsic_critics
        ]

        # method of gradient aggregation e,g., softmax vs argmax
        self.weight_option = weight_option
        # for logging which index of intrinsic reward is most contributing for extrinsic optimality
        self.contributing_indices = [
            str(i) for i in range(self.intrinsic_reward_fn.num_rewards)
        ]
        # for logging how much improvement of intrinsic rewards during inner-level updates
        self.init_avg_intrinsic_rewards = {}
        self.final_avg_intrinsic_rewards = {}
        self.probability_history = np.array(
            [0.0 for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )

        # ensure the actor is in desireable dtype and device
        self.wall_clock_time = 0
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(
        self, env: gym.Env, sampler: OnlineSampler, seed: int, learning_progress: float
    ):
        # === Set it to training mode === #
        self.train()

        # === Initialize the logging parameters === #
        total_timesteps, total_sample_time, total_update_time = 0, 0, 0
        policy_dict, gradient_dict = {}, {}

        # === initialize the policy dict with outer-level policy === #
        for i in range(self.intrinsic_reward_fn.num_rewards):
            actor_idx = f"{i}_{0}"
            policy_dict[actor_idx] = deepcopy(self.actor)

        # === SAMPLE FOR INITIAL UPDATE === #
        init_batch, sample_time = sampler.collect_samples(env, self.actor, seed)
        self.actor.record_state_visitations(init_batch["states"], alpha=1.0)
        total_timesteps += init_batch["states"].shape[0]

        # === UPDATE BASE CRITIC AND GET TRUE GRADIENTS === #
        # base_gradients = self.learn_base_policy(init_batch)

        # === UPDATE VIA GRADIENT CHAIN === #
        loss_dict_list = []
        for i in range(self.intrinsic_reward_fn.num_rewards):
            for j in range(self.num_inner_updates):
                # === Identify this is inner or outer-level update === #
                prefix = "outer" if j == self.num_inner_updates - 1 else "inner"

                # === Define actor === #
                actor_idx = f"{i}_{j}"
                future_actor_idx = f"{i}_{j+1}"
                actor = policy_dict[actor_idx]

                # === Sample batch for corresponding actor === #
                if prefix == "inner":
                    if j == 0:
                        batch = deepcopy(init_batch)
                        sample_time = 0
                        timesteps = 0
                    else:
                        batch, sample_time = sampler.collect_samples(env, actor, seed)
                        timesteps = batch["states"].shape[0]
                else:
                    batch, sample_time = sampler.collect_samples(env, actor, seed)
                    actor.record_state_visitations(batch["states"], alpha=1.0)
                    timesteps = batch["states"].shape[0]

                # === Update probability history using exponential moving average === #
                beta = 0.95
                self.probability_history[i] = (
                    beta * self.probability_history[i]
                    + (1 - beta) * batch["rewards"].mean()
                )

                # === Perform an inner-level update === #
                (
                    loss_dict,
                    update_time,
                    actor_clone,
                    gradients,
                    avg_intrinsic_rewards,
                ) = self.learn_exploratory_policy(actor, batch, i, prefix)

                # === Logging the outputs of inner-level update === #
                # This is important as it retains the actor with computational graph
                # that will be used to backpropagate the gradients
                loss_dict_list.append(loss_dict)

                gradient_dict[actor_idx] = gradients
                policy_dict[future_actor_idx] = actor_clone

                # === Logging for intrinsic reward improvement === #
                if j == 0:
                    self.init_avg_intrinsic_rewards[str(i)] = avg_intrinsic_rewards
                elif j == self.num_inner_updates - 1:
                    self.final_avg_intrinsic_rewards[str(i)] = avg_intrinsic_rewards

                # === accumulate the total timespteps and times for fairness === #
                total_timesteps += timesteps
                total_sample_time += sample_time
                total_update_time += update_time

        # === Backpropagation === #
        outer_gradients = []
        for i in range(self.intrinsic_reward_fn.num_rewards):
            gradients = gradient_dict[f"{i}_{self.num_inner_updates - 1}"]
            for j in reversed(range(self.num_inner_updates - 1)):
                iter_idx = f"{i}_{j}"
                Hv = grad(
                    gradient_dict[iter_idx],
                    policy_dict[iter_idx].parameters(),
                    grad_outputs=gradients,
                )
                gradients = tuple(
                    g - self.inner_actor_lr * h for g, h in zip(gradients, Hv)
                )
            outer_gradients.append(gradients)

        # === Identify the most contributing index for logging === #
        most_contributing_index = np.argmax(self.probability_history)

        # === Group by parameter=== #
        outer_gradients_transposed = list(zip(*outer_gradients))

        # === Gradient aggregation === #
        if self.weight_option == "argmax":
            weights = np.zeros_like(self.probability_history)
            weights[np.argmax(self.probability_history)] = 1.0
        elif self.weight_option == "softmax":
            weights = self.probability_history
            weights = weights / (weights.sum() + 1e-8)

        gradients = tuple(
            sum(w * g for w, g in zip(weights, grads_per_param))
            for grads_per_param in outer_gradients_transposed
        )

        # === convex combination of IRPO gradient with true gradient === #
        irm_weight = self.irm_weight * learning_progress
        # gradients = tuple(
        #     (1 - irm_weight) * g + irm_weight * true_g
        #     for g, true_g in zip(gradients, base_gradients)
        # )

        # === Perform outer-level update === #
        backtrack_iter, backtrack_success = self.learn_outer_level_polocy(
            states=init_batch["states"],
            grads=gradients,
        )

        # === Logging a visitation maps of the base policy === #
        visitation_dict = {}
        visitation_dict["visitation map (outer)"] = self.actor.state_visitation
        for i in range(self.intrinsic_reward_fn.num_rewards):
            idx = f"{i}_{self.num_inner_updates}"
            name = f"visitation map ({self.contributing_indices[i]})"
            visitation_dict[name] = policy_dict[idx].state_visitation

        # === Make dictionary for logging === #
        self.wall_clock_time += total_sample_time + total_update_time
        loss_dict = self.average_dict_values(loss_dict_list)
        loss_dict[f"{self.name}/analytics/avg_extrinsic_rewards"] = init_batch[
            "rewards"
        ].mean()
        loss_dict[f"{self.name}/analytics/wall_clock_time (hr)"] = (
            self.wall_clock_time / 3600.0
        )
        loss_dict[f"{self.name}/analytics/sample_time"] = total_sample_time
        loss_dict[f"{self.name}/analytics/update_time"] = total_update_time
        loss_dict[f"{self.name}/parameters/num vectors"] = (
            self.intrinsic_reward_fn.num_rewards
        )
        loss_dict[f"{self.name}/parameters/num_inner_updates"] = self.num_inner_updates
        loss_dict[f"{self.name}/analytics/Contributing Option"] = int(
            self.contributing_indices[most_contributing_index]
        )
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl
        loss_dict[f"{self.name}/analytics/inner_actor_lr"] = self.inner_actor_lr
        loss_dict[f"{self.name}/analytics/irm_weight"] = irm_weight

        loss_dict.update(self.intrinsic_reward_fn.loss_dict)

        for i in range(self.intrinsic_reward_fn.num_rewards):
            intrinsic_avg_rewards_improvement = (
                self.final_avg_intrinsic_rewards[str(i)]
                - self.init_avg_intrinsic_rewards[str(i)]
            )
            loss_dict[f"{self.name}/analytics/R_int_improvement ({i})"] = (
                intrinsic_avg_rewards_improvement
            )

        return loss_dict, total_timesteps, visitation_dict

    def learn_base_policy(self, batch: dict):
        # === Set it to training mode === #
        self.train()

        # === Prepare ingredients === #
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        # === Compute advantages and returns === #
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # === Prepare for critic learning === #
        batch_size = states.shape[0]
        critic_iteration = 5
        mb_size = batch_size // critic_iteration
        perm = torch.randperm(batch_size)

        # === Learn a base critic === #
        critic_losses = []
        for j in range(critic_iteration):
            indices = perm[j * mb_size : (j + 1) * mb_size]

            critic_loss = self.critic_loss(
                self.critic, states[indices], returns[indices]
            )
            critic_losses.append(critic_loss.item())

            self.base_critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.base_critic_optimizer.step()

        critic_loss = sum(critic_losses) / len(critic_losses)

        # === Learn exploratory policy === #
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss, entropy_loss = self.actor_loss(
            self.actor, states, actions, old_logprobs, advantages
        )

        loss = actor_loss - entropy_loss

        # Compute gradients
        gradients = torch.autograd.grad(loss, self.actor.parameters())
        gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        return gradients

    def learn_exploratory_policy(
        self, actor: nn.Module, batch: dict, i: int, prefix: str
    ):
        # === Set it to training mode === #
        self.train()

        # === Initialize the time === #
        t0 = time.time()

        # === Prepare ingredients === #
        states = self.preprocess_state(batch["states"])
        next_states = self.preprocess_state(batch["next_states"])
        actions = self.preprocess_state(batch["actions"])
        extrinsic_rewards = self.preprocess_state(batch["rewards"])
        intrinsic_rewards, source = self.intrinsic_reward_fn(states, next_states, i)
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        # === Compute advantages and returns === #
        with torch.no_grad():
            extrinsic_values = self.extrinsic_critics[i](states)
            intrinsic_values = self.intrinsic_critics[i](states)

            extrinsic_advantages, extrinsic_returns = estimate_advantages(
                extrinsic_rewards,
                terminals,
                extrinsic_values,
                gamma=self.gamma,
                gae=self.gae,
            )
            intrinsic_advantages, intrinsic_returns = estimate_advantages(
                intrinsic_rewards,
                terminals,
                intrinsic_values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # === Prepare for critic learning === #
        batch_size = states.shape[0]
        critic_iteration = 5
        mb_size = batch_size // critic_iteration
        perm = torch.randperm(batch_size)

        # === Learn extrinsic critic === #
        extrinsic_critic = self.extrinsic_critics[i]
        extrinsic_optim = self.extrinsic_critic_optimizers[i]

        extrinsic_losses = []
        for j in range(critic_iteration):
            # randomly sample mini-batch
            indices = perm[j * mb_size : (j + 1) * mb_size]
            critic_loss = self.critic_loss(
                extrinsic_critic, states[indices], extrinsic_returns[indices]
            )
            extrinsic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(extrinsic_critic.parameters(), max_norm=0.5)
            extrinsic_optim.step()
            extrinsic_losses.append(critic_loss.item())

        extrinsic_critic_loss = sum(extrinsic_losses) / len(extrinsic_losses)

        # === Learn intrinsic critic === #
        intrinsic_critic = self.intrinsic_critics[i]
        intrinsic_optim = self.intrinsic_critic_optimizers[i]

        intrinsic_losses = []
        for j in range(critic_iteration):
            indices = perm[j * mb_size : (j + 1) * mb_size]
            critic_loss = self.critic_loss(
                intrinsic_critic, states[indices], intrinsic_returns[indices]
            )
            intrinsic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(intrinsic_critic.parameters(), max_norm=0.5)
            intrinsic_optim.step()
            intrinsic_losses.append(critic_loss.item())

        intrinsic_critic_loss = sum(intrinsic_losses) / len(intrinsic_losses)

        # === Learn exploratory policy === #
        actor_clone = deepcopy(actor)
        advantages = extrinsic_advantages if prefix == "outer" else intrinsic_advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss, entropy_loss = self.actor_loss(
            actor, states, actions, old_logprobs, advantages
        )

        if prefix == "outer":
            # include the entropy loss in the outer-level policy update
            loss = actor_loss - entropy_loss
        else:
            # do not include the entropy loss in the inner-level policy update
            # this is because intrinsic reward is already abundant
            loss = actor_loss

        # Compute gradients
        gradients = torch.autograd.grad(loss, actor.parameters(), create_graph=True)
        gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        # Manual SGD update (structured, not flat)
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.inner_actor_lr * g

        # Logging
        actor_grad_norm = torch.sqrt(
            sum(g.pow(2).sum() for g in gradients if g is not None)
        )

        # Update of intrinsic reward if necessary (DRND only)
        self.intrinsic_reward_fn.learn(states, next_states, i, source)

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/extrinsic_critic_loss": extrinsic_critic_loss,
            f"{self.name}/loss/intrinsic_critic_loss": intrinsic_critic_loss,
            f"{self.name}/grad/actor": actor_grad_norm.item(),
            f"{self.name}/analytics/avg_intrinsic_rewards ({self.contributing_indices[i]})": torch.mean(
                intrinsic_rewards
            ).item(),
        }
        norm_dict = self.compute_weight_norm(
            [actor],
            ["actor"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)

        self.eval()
        update_time = time.time() - t0

        return (
            loss_dict,
            update_time,
            actor_clone,
            gradients,
            torch.mean(intrinsic_rewards).item(),
        )

    def learn_outer_level_polocy(
        self,
        states: np.ndarray,
        grads: tuple[torch.Tensor],
        damping: float = 1e-1,
        backtrack_iters: int = 15,
        backtrack_coeff: float = 0.7,
    ):
        if self.outer_level_update_mode == "trpo":
            states = self.preprocess_state(states)

            old_actor = deepcopy(self.actor)

            # Flatten meta-gradients
            meta_grad_flat = torch.cat([g.view(-1) for g in grads]).detach()

            # KL function (closure)
            def kl_fn():
                return compute_kl(old_actor, self.actor, states)

            # Define HVP function
            Hv = lambda v: hessian_vector_product(kl_fn, self.actor, damping, v)

            # Compute step direction with CG
            step_dir = conjugate_gradients(Hv, meta_grad_flat, nsteps=10)

            # Compute step size to satisfy KL constraint
            sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
            lm = torch.sqrt(sAs / self.target_kl)
            full_step = step_dir / (lm + 1e-8)

            # Apply update
            with torch.no_grad():
                old_params = flat_params(self.actor)

                # Backtracking line search
                success = False
                for i in range(backtrack_iters):
                    step_frac = backtrack_coeff**i
                    new_params = old_params - step_frac * full_step
                    set_flat_params(self.actor, new_params)
                    kl = compute_kl(old_actor, self.actor, states)

                    if kl <= self.target_kl:
                        success = True
                        break

                if not success:
                    set_flat_params(self.actor, old_params)

            return i, success
        elif self.outer_level_update_mode == "sgd":
            # SGD update
            with torch.no_grad():
                for p, g in zip(self.actor.parameters(), grads):
                    p -= self.outer_actor_lr * g

            return 0, True

    def actor_loss(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ):
        # === Naive policy gradient loss === #
        _, metaData = actor(states)
        logprobs = actor.log_prob(metaData["dist"], actions)
        entropy = actor.entropy(metaData["dist"])
        # the ratio should be just one as this is a one-batch update,
        # but used here for numerical stability
        ratios = torch.exp(logprobs - old_logprobs)

        actor_loss = -(ratios * advantages).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        return actor_loss, entropy_loss

    def critic_loss(
        self, critic: nn.Module, states: torch.Tensor, returns: torch.Tensor
    ):
        values = critic(states)
        # MSE loss for critic is important as we want to emphasize the outliers
        # where the outliers are rarely achieved reward in sparse-reward environments
        value_loss = self.mse_loss(values, returns)

        return value_loss
