from typing import Optional, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[list[int], tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        initialization: str = "default",
        dropout_rate: Optional[float] = None,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        model = []

        # Example: Initialization for a layer based on activation function
        if activation == nn.ReLU():
            gain = nn.init.calculate_gain("relu")
        elif activation == nn.LeakyReLU():
            gain = nn.init.calculate_gain("leaky_relu")
        elif activation == nn.Tanh():
            gain = nn.init.calculate_gain("tanh")
        elif activation == nn.Sigmoid():
            gain = nn.init.calculate_gain("sigmoid")
        elif activation == nn.ELU():
            gain = nn.init.calculate_gain("elu")
        elif activation == nn.Softplus():
            gain = nn.init.calculate_gain("softplus")
        elif activation == nn.Softsign():
            gain = nn.init.calculate_gain("softsign")
        else:
            gain = 1.0  # Default if no known activation matches

        # Initialize hidden layers
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.01)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=1.414)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=1.414)
                linear_layer.bias.data.fill_(0.0)

            model += (
                [linear_layer, activation] if activation is not None else [linear_layer]
            )

            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]

        # Initialize output layer
        if output_dim is not None:
            linear_layer = nn.Linear(hidden_dims[-1], output_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=0.01)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=1)
                linear_layer.bias.data.fill_(0.0)

            model += [linear_layer]
            self.output_dim = output_dim

        self.model = nn.Sequential(*model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
