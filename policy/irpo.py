import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """
    Intrinsic Reward Policy Optimization (IRPO) Learner.

    This class implements a bi-level optimization framework:
    1. Exploratory Phase (Inner Loop): Generates 'exploratory policies' by optimizing purely
       for specific intrinsic rewards.
    2. Base Policy Phase (Outer Loop): Updates the 'base policy' to maximize extrinsic reward
       by aggregating gradients from the most promising exploratory policies.
    """

    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        intrinsic_reward_fn: IntrinsicRewardFunctions,
        timesteps: int,
        num_exploratory_updates: int,
        base_policy_update_mode: str = "trpo",
        base_actor_lr: float = 3e-4,
        exploratory_actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_scaler: float = 1e-3,
        target_kl: float = 0.03,
        l2_reg: float = 1e-8,
        gamma: float = 0.99,
        gae: float = 0.95,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        self.name = "IRPO"
        self.device = device

        # Policy and Environment parameters
        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.timesteps = timesteps

        # IRPO Optimization parameters
        self.base_policy_update_mode = (
            base_policy_update_mode  # "trpo" is recommended for stability
        )
        self.num_exploratory_updates = num_exploratory_updates

        # Learning rates
        self.base_actor_lr = base_actor_lr  # Only used if base mode is SGD
        self.exploratory_actor_lr = exploratory_actor_lr  # Used for exploratory loop
        self.critic_lr = critic_lr

        # Hyperparameters
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.target_kl = target_kl  # Constraint for TRPO
        self.l2_reg = l2_reg

        # Neural Networks
        self.actor = actor  # The "Base" policy
        self.critic = critic
        self.intrinsic_reward_fn = intrinsic_reward_fn

        # Critics Setup:
        # We need separate critics for every intrinsic reward type to correctly estimate
        # advantages for the specific intrinsic objective.
        self.extrinsic_critics = nn.ModuleList(
            [deepcopy(self.critic) for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )
        self.intrinsic_critics = nn.ModuleList(
            [deepcopy(self.critic) for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )

        # Optimizers for the critics
        self.extrinsic_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.extrinsic_critics
        ]
        self.intrinsic_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            for critic in self.intrinsic_critics
        ]

        # Tracking contributing (high reward-inducing) intrinsic rewards
        self.contributing_indices = [
            str(i) for i in range(self.intrinsic_reward_fn.num_rewards)
        ]

        # Tracking metrics for intrinsic reward performance
        self.init_avg_intrinsic_rewards = {}
        self.final_avg_intrinsic_rewards = {}

        # Probability history: tracks which intrinsic reward leads to best extrinsic returns.
        # Initialized to 0, updated via EMA.
        self.probability_history = np.array(
            [0.0 for _ in range(self.intrinsic_reward_fn.num_rewards)]
        )

        # Storage for the best policies found during exploration (for inference/eval)
        self.Nth_exploratory_policies = {
            f"{i}": deepcopy(actor) for i in range(self.intrinsic_reward_fn.num_rewards)
        }

        self.wall_clock_time = 0
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        """
        Standard forward pass.
        If deterministic (Eval mode), it selects the 'best' exploratory policy found so far
        instead of the base policy.
        """
        state = self.preprocess_state(state)
        if deterministic:
            # Select the policy associated with the highest historical return
            actor = self.Nth_exploratory_policies[
                str(np.argmax(self.probability_history))
            ]
            a, metaData = actor(state, deterministic=deterministic)
        else:
            a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def compute_weights(self, learning_progress, epsilon=1e-6):
        """
        probability_history: torch.Tensor, shape [num_actions]
        learning_progress: float in [0, 1]
        """
        # === 1. Define the Horizon (10% of training) === #
        target_horizon = 0.10

        # Normalize progress: 0.0 (start) -> 1.0 (at 10% progress)
        # We clamp it to max 1.0 to handle the "else" case safely
        progress_ratio = min(learning_progress / target_horizon, 1.0)

        probability_history = torch.tensor(self.probability_history).to(self.device)

        # === 2. Check if we passed the 10% threshold === #
        if progress_ratio >= 1.0:
            # === Phase 2: Exploitation (After 10%) === #
            # Use hard argmax (Temperature effectively 0)
            weights = torch.zeros_like(probability_history)
            weights[torch.argmax(probability_history)] = 1.0
            self.temperature = 0.0  # For logging

        else:
            # === Phase 1: Exploration (First 10%) === #
            # We want temperature to go from 1.0 -> 0.0

            # Linear Decay:
            self.temperature = 1.0 - progress_ratio + epsilon

            # Apply Softmax with decaying temperature
            # High temp (1.0) = Soft weights (Explore)
            # Low temp (0.1) = Sharp weights (Exploit)
            weights = F.softmax(probability_history / self.temperature, dim=0)

        return weights

    def learn(
        self, env: gym.Env, sampler: OnlineSampler, seed: int, learning_progress: float
    ):
        """
        Main bi-level optimization loop.
        1. Exploratory Phase: Updates copies of the actor on intrinsic rewards.
        2. Backprop: Computes meta-gradients through the exploratory updates.
        3. Base Policy Phase: Updates the base actor on extrinsic objectives.
        """
        self.train()

        total_timesteps, total_sample_time, total_update_time = 0, 0, 0
        policy_dict, gradient_dict = {}, {}

        # Snapshot the current base policy to start the exploratory loop
        for i in range(self.intrinsic_reward_fn.num_rewards):
            actor_idx = f"{i}_{0}"
            policy_dict[actor_idx] = deepcopy(self.actor)

        # 1. Collect initial data with the BASE policy (for comparison)
        init_batch, sample_time = sampler.collect_samples(env, self.actor, seed)
        self.actor.record_state_visitations(init_batch["states"], alpha=1.0)
        total_timesteps += init_batch["states"].shape[0]

        loss_dict_list = []

        # 2. EXPLORATORY PHASE: Iterate over each intrinsic reward type
        for i in range(self.intrinsic_reward_fn.num_rewards):
            for j in range(self.num_exploratory_updates):
                # Identify if this is a standard exploratory update or the 'base' transition point
                # (Note: 'base' prefix here denotes the step used for the meta-update calculation)
                prefix = (
                    "base" if j == self.num_exploratory_updates - 1 else "exploratory"
                )

                # Retrieve current step's policy
                actor_idx = f"{i}_{j}"
                future_actor_idx = f"{i}_{j+1}"
                actor = policy_dict[actor_idx]

                # Sample data: Use Init batch for step 0, new samples for subsequent steps
                if prefix == "exploratory":
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

                # Update historical performance (EMA) based on Extrinsic Reward obtained
                beta = 0.9995
                self.probability_history[i] = (
                    beta * self.probability_history[i]
                    + (1 - beta) * batch["rewards"].mean()
                )

                # Perform Gradient Descent Step (Exploratory Update)
                # This returns the gradients and a cloned, updated actor
                (
                    loss_dict,
                    update_time,
                    actor_clone,
                    gradients,
                    avg_intrinsic_rewards,
                ) = self.learn_exploratory_policy(actor, batch, i, prefix)

                # Store results for the meta-backward pass
                loss_dict_list.append(loss_dict)
                gradient_dict[actor_idx] = gradients
                policy_dict[future_actor_idx] = (
                    actor_clone  # Store updated policy for next step
                )

                # Log intrinsic reward improvements
                if j == 0:
                    self.init_avg_intrinsic_rewards[str(i)] = avg_intrinsic_rewards
                elif j == self.num_exploratory_updates - 1:
                    self.final_avg_intrinsic_rewards[str(i)] = avg_intrinsic_rewards

                total_timesteps += timesteps
                total_sample_time += sample_time
                total_update_time += update_time

        # 3. META-GRADIENT COMPUTATION (Backpropagation through time)
        # We need to compute how the initial parameters affect the final outcome.
        # This involves Hessian-Vector Products (HVP).
        outer_gradients = []
        for i in range(self.intrinsic_reward_fn.num_rewards):
            # Start with gradients from the last exploratory step
            gradients = gradient_dict[f"{i}_{self.num_exploratory_updates - 1}"]

            # Walk backward from last step to 0
            for j in reversed(range(self.num_exploratory_updates - 1)):
                iter_idx = f"{i}_{j}"

                # Compute (I - lr * Hessian) * gradients using HVP
                Hv = grad(
                    gradient_dict[iter_idx],
                    policy_dict[iter_idx].parameters(),
                    grad_outputs=gradients,
                )
                gradients = tuple(
                    g - self.exploratory_actor_lr * h for g, h in zip(gradients, Hv)
                )
            outer_gradients.append(gradients)

        most_contributing_index = np.argmax(self.probability_history)

        # 4. GRADIENT AGGREGATION
        # Combine gradients from different intrinsic directions using computed weights
        weights = self.compute_weights(learning_progress).cpu().numpy()

        outer_gradients_transposed = list(zip(*outer_gradients))
        gradients = tuple(
            sum(w * g for w, g in zip(weights, grads_per_param))
            for grads_per_param in outer_gradients_transposed
        )

        # 5. BASE POLICY UPDATE (Outer Loop)
        # Apply the aggregated meta-gradient to the base policy using TRPO or SGD
        backtrack_iter, backtrack_success = self.learn_base_policy(
            states=init_batch["states"],
            grads=gradients,
        )

        # Logging / Visualization preparation
        visitation_dict = {}
        visitation_dict["visitation map (base)"] = self.actor.state_visitation
        for i in range(self.intrinsic_reward_fn.num_rewards):
            idx = f"{i}_{self.num_exploratory_updates}"
            name = f"visitation map ({self.contributing_indices[i]})"
            visitation_dict[name] = policy_dict[idx].state_visitation

        # Dictionary construction for logger
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
        loss_dict[f"{self.name}/parameters/num_exploratory_updates"] = (
            self.num_exploratory_updates
        )
        loss_dict[f"{self.name}/analytics/Contributing Option"] = int(
            self.contributing_indices[most_contributing_index]
        )
        loss_dict[f"{self.name}/analytics/Backtrack_iter"] = backtrack_iter
        loss_dict[f"{self.name}/analytics/Backtrack_success"] = backtrack_success
        loss_dict[f"{self.name}/analytics/target_kl"] = self.target_kl
        loss_dict[f"{self.name}/analytics/exploratory_actor_lr"] = (
            self.exploratory_actor_lr
        )
        loss_dict[f"{self.name}/analytics/temperature"] = self.temperature
        loss_dict.update(self.intrinsic_reward_fn.loss_dict)

        for i in range(self.intrinsic_reward_fn.num_rewards):
            improvement = (
                self.final_avg_intrinsic_rewards[str(i)]
                - self.init_avg_intrinsic_rewards[str(i)]
            )
            loss_dict[f"{self.name}/analytics/R_int_improvement ({i})"] = improvement

        return loss_dict, total_timesteps, visitation_dict

    def learn_exploratory_policy(
        self, actor: nn.Module, batch: dict, i: int, prefix: str
    ):
        """
        Performs a single exploratory update.
        Calculates intrinsic rewards, updates critics, and performs a differentiable actor update.
        """
        self.train()
        t0 = time.time()

        # Preprocessing data
        states = self.preprocess_state(batch["states"])
        next_states = self.preprocess_state(batch["next_states"])
        actions = self.preprocess_state(batch["actions"])
        extrinsic_rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])

        # Calculate Intrinsic Reward for the i-th objective
        intrinsic_rewards, source = self.intrinsic_reward_fn(states, next_states, i)

        # Estimate Advantages
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
            # Intrinsic returns often use gamma=1.0 as they are dense per-step signals
            intrinsic_advantages, intrinsic_returns = estimate_advantages(
                intrinsic_rewards,
                terminals,
                intrinsic_values,
                gamma=1.0,
                gae=self.gae,
            )

        # Update Critics (MSE Loss)
        # We iterate multiple times to ensure the value function is accurate
        batch_size = states.shape[0]
        critic_iteration = 5
        mb_size = batch_size // critic_iteration
        perm = torch.randperm(batch_size)

        # 1. Update Extrinsic Critic
        extrinsic_critic = self.extrinsic_critics[i]
        extrinsic_optim = self.extrinsic_critic_optimizers[i]
        extrinsic_losses = []
        for j in range(critic_iteration):
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

        # 2. Update Intrinsic Critic
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

        # 3. Update Actor (Exploratory Policy)
        actor_clone = deepcopy(actor)

        # Select advantage based on whether we are in the 'exploratory' (intrinsic)
        # or 'base' (extrinsic) phase of the loop for this specific calculation.
        advantages = extrinsic_advantages if prefix == "base" else intrinsic_advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss, entropy_loss = self.actor_loss(actor, states, actions, advantages)

        # Only add entropy regularization for base policy updates to encourage exploration there,
        # exploratory updates are already driven by intrinsic rewards.
        if prefix == "base":
            loss = actor_loss - entropy_loss
        else:
            loss = actor_loss

        # Calculate Gradients with create_graph=True
        # This is CRITICAL for meta-learning: allows backprop through this gradient step later.
        gradients = torch.autograd.grad(loss, actor.parameters(), create_graph=True)
        gradients = self.clip_grad_norm(gradients, max_norm=0.5)

        # Manual SGD update on the clone to keep the graph connected
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.exploratory_actor_lr * g

        # If this is the final exploratory step, save this policy as a potential inference policy
        if prefix == "base":
            self.Nth_exploratory_policies[str(i)] = actor_clone

        # Update Intrinsic Reward Generator (if it has learnable parameters, e.g., DRND)
        self.intrinsic_reward_fn.learn(states, next_states, i, source)

        # Stats & Logging
        actor_grad_norm = torch.sqrt(
            sum(g.pow(2).sum() for g in gradients if g is not None)
        )

        loss_dict = {
            f"{self.name}/loss/actor_loss": actor_loss.item(),
            f"{self.name}/loss/extrinsic_critic_loss": extrinsic_critic_loss,
            f"{self.name}/loss/intrinsic_critic_loss": intrinsic_critic_loss,
            f"{self.name}/grad/actor": actor_grad_norm.item(),
            f"{self.name}/analytics/avg_intrinsic_rewards ({self.contributing_indices[i]})": torch.mean(
                intrinsic_rewards
            ).item(),
        }

        # Log weight norms for debugging explosion/vanishing issues
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

    def learn_base_policy(
        self,
        states: np.ndarray,
        grads: tuple[torch.Tensor],
        damping: float = 1e-1,
        backtrack_iters: int = 15,
        backtrack_coeff: float = 0.7,
    ):
        """
        Updates the base policy using the aggregated meta-gradients.
        Standard TRPO implementation:
        1. Conjugate Gradient to find step direction (using Hessian-Vector Products).
        2. Line Search to ensure KL constraint is met.
        """
        if self.base_policy_update_mode == "trpo":
            states = self.preprocess_state(states)
            old_actor = deepcopy(self.actor)

            # Flatten the aggregated meta-gradients
            meta_grad_flat = torch.cat([g.view(-1) for g in grads]).detach()

            # KL divergence closure for Hessian Vector Product
            def kl_fn():
                return compute_kl(old_actor, self.actor, states)

            Hv = lambda v: hessian_vector_product(kl_fn, self.actor, damping, v)

            # Compute search direction (F_inv * g) via CG
            step_dir = conjugate_gradients(Hv, meta_grad_flat, nsteps=10)

            # Compute step size scaling (Lagrange multiplier)
            sAs = 0.5 * torch.dot(step_dir, Hv(step_dir))
            lm = torch.sqrt(sAs / self.target_kl)
            full_step = step_dir / (lm + 1e-8)

            # Line Search
            with torch.no_grad():
                old_params = flat_params(self.actor)
                success = False
                for i in range(backtrack_iters):
                    step_frac = backtrack_coeff**i
                    new_params = old_params - step_frac * full_step
                    set_flat_params(self.actor, new_params)

                    # Verify KL constraint
                    kl = compute_kl(old_actor, self.actor, states)
                    if kl <= self.target_kl:
                        success = True
                        break

                if not success:
                    set_flat_params(self.actor, old_params)

            return i, success

        elif self.base_policy_update_mode == "sgd":
            # Simple fallback: vanilla Gradient Descent
            with torch.no_grad():
                for p, g in zip(self.actor.parameters(), grads):
                    p -= self.base_actor_lr * g
            return 0, True

    def actor_loss(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """Standard Policy Gradient Loss with Entropy Regularization."""
        _, metaData = actor(states)
        logprobs = actor.log_prob(metaData["dist"], actions)
        entropy = actor.entropy(metaData["dist"])

        actor_loss = -(logprobs * advantages).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        return actor_loss, entropy_loss

    def critic_loss(
        self, critic: nn.Module, states: torch.Tensor, returns: torch.Tensor
    ):
        """Standard MSE Value Loss."""
        values = critic(states)
        value_loss = self.mse_loss(values, returns)
        return value_loss
