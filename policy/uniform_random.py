import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.layers.base import Base

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.rl import estimate_advantages

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class UniformRandom(Base):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        is_discrete: bool,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "UniformRandom"
        self.is_discrete = is_discrete
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        if self.is_discrete:
            logits = torch.rand(
                (1, self.action_dim), device=self.device, dtype=self.dtype
            )
            a = torch.argmax(logits, dim=-1)
            a = F.one_hot(a, num_classes=logits.size(-1))
        else:
            # for continuous pull random to be ranged -1 to 1
            logits = (
                torch.rand((1, self.action_dim), device=self.device, dtype=self.dtype)
                * 2
                - 1
            )
            a = logits

        probs = torch.ones(1, device=self.device, dtype=self.dtype)
        logprobs = torch.zeros(1, device=self.device, dtype=self.dtype)
        entropy = torch.zeros(1, device=self.device, dtype=self.dtype)

        return a, {
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def learn(self, batch):
        pass
