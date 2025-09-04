import time

import numpy as np
import torch
import torch.nn as nn

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.rl import estimate_advantages


class PPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        nupdates: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        # constants
        self.name = "PPO"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.is_discrete = is_discrete

        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        self.nupdates = nupdates

        # trainable networks
        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_lambda
        )

        #
        self.to(self.dtype).to(self.device)

    def lr_lambda(self, step):
        return 1.0 - float(step) / float(self.nupdates)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        self.record_state_visitations(states)
        timesteps = states.shape[0]

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_actions = states[indices], actions[indices]
                mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # 1. Critic Loss (with optional regularization)
                value_loss = self.critic_loss(mb_states, mb_returns)
                # Track value loss for logging
                value_losses.append(value_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
                    mb_states, mb_actions, mb_old_logprobs, mb_advantages
                )

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                target_kl.append(kl_div.item())

                if kl_div.item() > self.target_kl:
                    break

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss
                losses.append(loss.item())

                # Update critic parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic],
                    ["actor", "critic"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        self.lr_scheduler.step()

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
            f"{self.name}/analytics/policy_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/analytics/critic_lr": self.optimizer.param_groups[1]["lr"],
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic],
            ["actor", "critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        self.eval()

        update_time = time.time() - t0

        return loss_dict, timesteps, update_time

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        _, metaData = self.actor(mb_states)
        logprobs = self.actor.log_prob(metaData["dist"], mb_actions)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self._entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def critic_loss(self, mb_states: torch.Tensor, mb_returns: torch.Tensor):
        mb_values = self.critic(mb_states)
        value_loss = self.mse_loss(mb_values, mb_returns)

        return value_loss
