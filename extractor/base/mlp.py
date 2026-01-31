import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from policy.layers.building_blocks import MLP


class NeuralNet(nn.Module):
    def __init__(
        self,
        state_dim: tuple,
        feature_dim: int,
        encoder_fc_dim: list = [512, 256, 256, 128],
        activation: nn.Module = nn.Tanh(),
        device: torch.device = torch.device("cpu"),
    ):
        super(NeuralNet, self).__init__()

        # Parameters
        self.state_dim = state_dim
        self.input_dim = np.prod(state_dim)
        self.feature_dim = feature_dim
        self.encoder_fc_dim = encoder_fc_dim
        self.device = device

        ### Check the validity of the input dimensions
        if len(encoder_fc_dim) < 1:
            raise ValueError(
                "encoder_fc must have at least one element for the first layer."
            )

        self.logstd_range = (-5, 2)

        # Activation functions
        self.act = activation

        ### Encoding module
        self.encoder = MLP(
            input_dim=self.input_dim,
            hidden_dims=encoder_fc_dim,
            output_dim=feature_dim,
            activation=self.act,
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor, deterministic: bool = True):
        if len(state.shape) > 2:
            raise ValueError(
                f"The state representation is positional. Shound be 2D, given {len(state.shape)}D state dimension."
            )
        feature = self.encoder(state)

        return feature, {}

    def decode(self, features: torch.Tensor, actions: torch.Tensor):
        return NotImplementedError
