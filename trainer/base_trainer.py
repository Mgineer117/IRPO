import os
from abc import abstractmethod

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from utils.sampler import OnlineSampler


# model-free policy trainer
class BaseTrainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        rendering: bool = False,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_max_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.rendering = rendering
        self.seed = seed

    @abstractmethod
    def train(self) -> dict[str, float]:
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def spectral_loss(
        self, policy: nn.Module, power_iters: int = 3, k: int = 2
    ) -> torch.Tensor:
        """Compute spectral loss of a given policy network.
        Loss → 0 when spectral norm of weight matrices = 1 and biases = 0.
        """
        with torch.no_grad():
            spectral_loss = 0.0
            device = next(policy.parameters()).device

            for name, param in policy.named_parameters():
                if "weight" in name:
                    W = param
                    A = W.T @ W

                    b_k = torch.randn(A.shape[1], device=device)
                    for _ in range(power_iters):
                        b_k1 = A @ b_k
                        b_k1_norm = torch.norm(b_k1) + 1e-12
                        b_k = b_k1 / b_k1_norm

                    sigma = torch.norm(W @ b_k)  # spectral norm approximation
                    spectral_loss += (sigma**k - 1) ** 2

                elif "bias" in name:
                    spectral_loss += torch.sum(param ** (2 * k))

        return spectral_loss.item()

    # power_iteration(np.array([[0.5, 0.5], [0.2, 0.8]]), 10)

    def discounted_return(self, rewards, gamma):
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(
        self, image: np.ndarray | plt.Figure, step: int, logdir: str, name: str
    ):
        image_list = [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    @abstractmethod
    def save_model(self, e):
        pass

    def visitation_to_rgb(self, visitation_map: np.ndarray) -> np.ndarray:
        visitation_map = np.squeeze(visitation_map)  # Make sure it's 2D
        H, W = visitation_map.shape

        rgb_map = np.ones((H, W, 3), dtype=np.float32)  # Start with white

        # Zero visitation → gray
        zero_mask = visitation_map == 0
        rgb_map[zero_mask] = [0.5, 0.5, 0.5]

        # Nonzero visitation → white → blue gradient
        nonzero_mask = visitation_map > 0
        blue_intensity = visitation_map[nonzero_mask]

        rgb_map[nonzero_mask] = np.stack(
            [
                1.0 - blue_intensity,  # Red
                1.0 - blue_intensity,  # Green
                np.ones_like(blue_intensity),  # Blue
            ],
            axis=-1,
        )

        return rgb_map
