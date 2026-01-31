import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal

from policy.layers.base import Base
from policy.layers.building_blocks import MLP


class PPO_Actor(Base):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        is_discrete: bool,
        activation: nn.Module = nn.Tanh(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)

        self.is_discrete = is_discrete

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
        )
        if not self.is_discrete:
            self.logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        self.device = device
        self.to(self.device).to(self.dtype)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        """
        Forward pass for the actor network.
        :param state: Input state tensor.
        :param deterministic: If True, use deterministic action selection.
        :return: Action tensor and additional metadata.
        """
        state = self.preprocess_state(state)
        if self.is_discrete:
            a, metaData = self.discrete_forward(state, deterministic)
        else:
            a, metaData = self.continuous_forward(state, deterministic)

        return a, metaData

    def continuous_forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(state)

        ### Shape the output as desired
        mu = logits
        logstd = torch.clip(self.logstd, -5, 2)  # Clip logstd to avoid numerical issues
        std = torch.exp(logstd.expand_as(mu))
        dist = Normal(loc=mu, scale=std)

        a = dist.rsample()

        logprobs = dist.log_prob(a).unsqueeze(-1).sum(1)
        probs = torch.exp(logprobs)
        entropy = dist.entropy().sum(1)

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def discrete_forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(state)

        if deterministic:
            a = torch.argmax(logits, dim=-1)
            dist = None
            logprobs = torch.zeros_like(logits[:, 0:1])
            probs = torch.ones_like(logprobs)
            entropy = torch.zeros_like(logprobs)
        else:
            dist = Categorical(logits=logits)
            a = dist.sample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

            entropy = dist.entropy()

        a = F.one_hot(a, num_classes=logits.size(-1))
        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1).sum(1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        if self.is_discrete:
            return dist.entropy().unsqueeze(-1)
        else:
            return dist.entropy().unsqueeze(-1).sum(1)


class PPO_Critic(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(PPO_Critic, self).__init__()

        self.state_dim = np.prod(input_dim)

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            1,
            activation=activation,
            initialization="critic",
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
