import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.layers.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from utils.rl import estimate_advantages
from utils.wrapper import RunningMeanStd


class DRNDPPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        critic: PPO_Critic,
        is_discrete: bool,
        drnd_model: nn.Module,
        drnd_critic: PPO_Critic,
        positional_indices: list,
        nupdates: int | None = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        drnd_lr: float = 3e-4,
        num_minibatch: int = 8,
        minibatch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-8,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        ext_reward_scaler: float = 2.0,
        int_reward_scaler: float = 1.0,
        drnd_loss_scaler: float = 1.0,
        update_proportion: float = 0.25,
        alpha: float = 0.9,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "DRND_PPO"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.is_discrete = is_discrete

        self.positional_indices = positional_indices
        self.nupdates = nupdates
        self.num_minibatch = num_minibatch
        self.minibatch_size = minibatch_size
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.eps_clip = eps_clip
        self.ext_reward_scaler = ext_reward_scaler
        self.int_reward_scaler = int_reward_scaler
        self.drnd_loss_scaler = drnd_loss_scaler
        self.update_proportion = update_proportion
        self.alpha = alpha

        # trainable networks
        self.actor = actor
        self.critic = critic
        self.drnd = drnd_model
        self.drnd_critic = drnd_critic

        self.int_reward_rms = RunningMeanStd(shape=(1,))

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": actor_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
                {"params": self.drnd.parameters(), "lr": drnd_lr},
                {"params": self.drnd_critic.parameters(), "lr": critic_lr},
            ]
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_lambda
        )

        #
        self.to(self.dtype).to(self.device)

    def lr_lambda(self, step):
        if self.nupdates is not None:
            return 1.0 - float(step) / float(self.nupdates)
        else:
            return 1.0

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def intrinsic_reward(self, next_states: torch.Tensor):
        with torch.no_grad():
            predict_next_feature, target_next_feature = self.drnd(
                next_states[:, self.positional_indices]
            )

            mu = torch.mean(target_next_feature, axis=0)
            B2 = torch.mean(target_next_feature**2, axis=0)

            b1 = self.alpha * torch.sum(
                (predict_next_feature - mu) ** 2, dim=1, keepdim=True
            )
            b2 = (1 - self.alpha) * torch.sum(
                torch.sqrt(
                    torch.clip(
                        torch.abs(predict_next_feature**2 - mu**2) / (B2 - mu**2),
                        1e-6,
                        1,
                    )
                ),
                dim=-1,
                keepdim=True,
            )
            intrinsic_rewards = b1 + b2

            # normalize intrinsic rewards
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()
            self.int_reward_rms.update(intrinsic_rewards)

            intrinsic_rewards = intrinsic_rewards / (
                np.sqrt(self.int_reward_rms.var) + 1e-8
            )
            intrinsic_rewards = self.preprocess_state(intrinsic_rewards)

        return intrinsic_rewards

    def learn(self, batch):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        next_states = self.preprocess_state(batch["next_states"])
        actions = self.preprocess_state(batch["actions"])
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        ext_rewards = self.preprocess_state(batch["rewards"])
        int_rewards = self.intrinsic_reward(next_states)

        self.record_state_visitations(states)
        timesteps = states.shape[0]

        # Compute advantages and returns
        with torch.no_grad():
            ext_values = self.critic(states)
            ext_advantages, ext_returns = estimate_advantages(
                ext_rewards,
                terminals,
                ext_values,
                gamma=self.gamma,
                gae=self.gae,
            )

        # Compute advantages and returns
        with torch.no_grad():
            int_values = self.drnd_critic(states)
            int_advantages, int_returns = estimate_advantages(
                int_rewards,
                torch.zeros_like(terminals),  # No terminal for intrinsic rewards
                int_values,
                gamma=self.gamma,
                gae=self.gae,
            )

        advantages = (
            self.ext_reward_scaler * ext_advantages
            + self.int_reward_scaler * int_advantages
        )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track actor loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []
        drnd_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self.K):
            for n in range(self.num_minibatch):
                indices = torch.randperm(batch_size)[: self.minibatch_size]
                mb_states, mb_next_states = states[indices], next_states[indices]
                mb_actions, mb_old_logprobs = actions[indices], old_logprobs[indices]
                mb_ext_returns, mb_int_returns = (
                    ext_returns[indices],
                    int_returns[indices],
                )

                # advantages
                mb_advantages = advantages[indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # 1. Critic Loss (with optional regularization)
                value_loss = self.critic_loss(mb_states, mb_ext_returns, mb_int_returns)
                # Track value loss for logging
                value_losses.append(value_loss.item())

                # 2. actor Loss
                actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
                    mb_states, mb_actions, mb_old_logprobs, mb_advantages
                )

                # 3. DRND Loss
                drnd_loss = self.drnd_loss(next_states=mb_next_states)

                # Track actor loss for logging
                actor_losses.append(actor_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(clip_fraction)
                target_kl.append(kl_div.item())
                drnd_losses.append(drnd_loss.item())

                if kl_div.item() > self.target_kl:
                    break

                # Total loss
                loss = actor_loss - entropy_loss + 0.5 * value_loss + drnd_loss
                losses.append(loss.item())

                # Update critic parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                grad_dict = self.compute_gradient_norm(
                    [self.actor, self.critic, self.drnd],
                    ["actor", "critic", "drnd"],
                    dir=f"{self.name}",
                    device=self.device,
                )
                grad_dicts.append(grad_dict)
                self.optimizer.step()

            if kl_div.item() > self.target_kl:
                break

        self.lr_scheduler.step()
        if self.nupdates is not None:
            self.int_reward_scaler = 1 - np.clip(
                2 * (self.lr_scheduler.last_epoch / self.nupdates), 0.0, 1.0
            )

        # Logging
        loss_dict = {
            f"{self.name}/loss/loss": np.mean(losses),
            f"{self.name}/loss/actor_loss": np.mean(actor_losses),
            f"{self.name}/loss/value_loss": np.mean(value_losses),
            f"{self.name}/loss/entropy_loss": np.mean(entropy_losses),
            f"{self.name}/loss/drnd_loss": np.mean(drnd_losses),
            f"{self.name}/analytics/clip_fraction": np.mean(clip_fractions),
            f"{self.name}/analytics/klDivergence": target_kl[-1],
            f"{self.name}/analytics/K-epoch": k + 1,
            f"{self.name}/analytics/avg_rewards": torch.mean(ext_rewards).item(),
            f"{self.name}/analytics/int_rewards": torch.mean(int_rewards).item(),
            f"{self.name}/analytics/policy_lr": self.optimizer.param_groups[0]["lr"],
            f"{self.name}/analytics/critic_lr": self.optimizer.param_groups[1]["lr"],
            f"{self.name}/analytics/drnd_lr": self.optimizer.param_groups[2]["lr"],
            f"{self.name}/analytics/drnd_critic_lr": self.optimizer.param_groups[3][
                "lr"
            ],
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.actor, self.critic, self.drnd],
            ["actor", "critic", "drnd"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, ext_rewards, int_rewards, terminals, old_logprobs
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
        entropy_loss = self.entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def critic_loss(
        self,
        mb_states: torch.Tensor,
        mb_ext_returns: torch.Tensor,
        mb_int_returns: torch.Tensor,
    ):
        mb_ext_values = self.critic(mb_states)
        mb_int_values = self.drnd_critic(mb_states)

        ext_value_loss = self.mse_loss(mb_ext_values, mb_ext_returns)
        int_value_loss = self.mse_loss(mb_int_values, mb_int_returns)

        value_loss = ext_value_loss + int_value_loss

        return value_loss

    def drnd_loss(self, next_states: torch.Tensor):
        """Curiosity-driven(Distributional Random Network Distillation)"""
        predict_next_state_feature, target_next_state_feature = self.drnd(
            next_states[:, self.positional_indices]
        )
        idx = torch.randint(high=self.drnd.num_target, size=(next_states.shape[0],))
        with torch.no_grad():
            target = target_next_state_feature[
                idx, torch.arange(predict_next_state_feature.shape[0]), :
            ]
        forward_loss = self.mse_loss(predict_next_state_feature, target)

        # Proportion of exp used for predictor update
        mask = torch.rand(next_states.shape[0]).to(self.device)
        mask = mask < self.update_proportion

        forward_loss = (forward_loss * mask).sum() / torch.max(
            mask.sum(), torch.Tensor([1]).to(self.device)
        )

        return self.drnd_loss_scaler * forward_loss
