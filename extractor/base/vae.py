import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from policy.layers.building_blocks import MLP


class VAE(nn.Module):
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        feature_dim: int,
        encoder_fc_dim: list = [512, 256, 256, 128],
        decoder_fc_dim: list = [128, 256, 256, 512],
        activation: nn.Module = nn.Tanh(),
        device: torch.device = torch.device("cpu"),
    ):
        super(VAE, self).__init__()

        # Parameters
        self.state_dim = state_dim
        self.input_dim = np.prod(state_dim)
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.encoder_fc_dim = encoder_fc_dim
        self.decoder_fc_dim = decoder_fc_dim
        self.device = device

        ### Check the validity of the input dimensions
        if len(decoder_fc_dim) < 2:
            raise ValueError(
                "decoder_fc must have at least two elements: one for the first layer and one for the output layer."
            )
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
            activation=self.act,
        )

        self.mu = nn.Linear(
            in_features=encoder_fc_dim[-1],
            out_features=feature_dim,
        )
        self.logstd = nn.Linear(
            in_features=encoder_fc_dim[-1],
            out_features=feature_dim,
        )

        ### Decoding module
        self.de_latent = MLP(
            input_dim=feature_dim,
            hidden_dims=[int(decoder_fc_dim[0] / 2)],
            activation=self.act,
        )

        self.de_action = MLP(
            input_dim=action_dim,
            hidden_dims=[int(decoder_fc_dim[0] / 2)],
            activation=self.act,
        )

        self.decoder = MLP(
            input_dim=decoder_fc_dim[0],
            hidden_dims=decoder_fc_dim[1:],
            output_dim=self.input_dim,
            activation=self.act,
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor, deterministic: bool = True):
        if len(state.shape) > 2:
            state = state.view(state.size(0), -1)
        out = self.encoder(state)
        mu = self.mu(out)
        logstd = torch.clamp(
            self.logstd(out),
            min=self.logstd_range[0],
            max=self.logstd_range[1],
        )
        std = torch.exp(logstd)

        if deterministic:
            feature = mu
        else:
            cov = torch.diag_embed(std**2)
            dist = MultivariateNormal(loc=mu, covariance_matrix=cov)

            feature = dist.rsample()

            # KL divergence loss
            kl = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=-1)
            kl_loss = 1e-4 * torch.mean(kl)

        return feature, {"loss": kl_loss}

    def decode(self, features: torch.Tensor, actions: torch.Tensor):
        out1 = self.de_latent(features)
        out2 = self.de_action(actions)

        out = torch.cat((out1, out2), axis=-1)
        reconstructed_state = self.decoder(out)

        return reconstructed_state.view(features.shape[0], *self.state_dim)
