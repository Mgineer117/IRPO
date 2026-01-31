import os
import pickle
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from utils.sampler import OnlineSampler


# model-free policy trainer
class ExtractorTrainer:
    def __init__(
        self,
        env: gym.Env,
        random_policy: Base,
        extractor: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        epochs: int = 1e6,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = random_policy
        self.extractor = extractor
        self.sampler = sampler

        self.logger = logger
        self.writer = writer

        # training parameters
        self.epochs = epochs

        # initialize the essential training components
        self.last_min_loss = 1e10

        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.loss_list = deque(maxlen=5)

        # Collect initial data
        # random_init_pos = True if self.extractor.name == "EigenOption" else False
        batch, _ = self.sampler.collect_samples(
            env=self.env, policy=self.policy, seed=self.seed, random_init_pos=True
        )

        # Train loop
        step = 0
        eval_idx = 0
        self.extractor.train()
        with tqdm(
            total=self.epochs, desc=f"{self.extractor.name} Training (Timesteps)"
        ) as pbar:
            while pbar.n < self.epochs:
                step = pbar.n + 1  # + 1 to avoid zero division

                loss_dict, _, eigenvalue_plot, update_time = self.extractor.learn(batch)

                # Calculate expected remaining time
                pbar.update(1)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / step
                remaining_time = avg_time_per_iter * (self.epochs - step)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.extractor.name}/analytics/epochs"] = step
                loss_dict[f"{self.extractor.name}/analytics/update_time"] = update_time
                loss_dict[f"{self.extractor.name}/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=step)

                #### EVALUATIONS ####
                if step > eval_idx * int(self.epochs / 1_000):

                    self.write_image(
                        image=eigenvalue_plot,
                        step=step,
                        logdir="Image",
                        name="Eigenvalues",
                    )

                    eval_idx += 1
                    self.loss_list.append(loss_dict[f"{self.extractor.name}/loss"])
                    if np.mean(self.loss_list) < self.last_min_loss:
                        self.save_model(step)

                torch.cuda.empty_cache()

        self.logger.print(
            "total extractor training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return step

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        if image is not None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            image_list = [image]
            path_image_path = os.path.join(logdir, name)
            self.logger.write_images(
                step=step, images=image_list, logdir=path_image_path
            )

    def write_video(self, rendering_imgs: list, step: int, logdir: str, name: str):
        path_render_path = os.path.join(logdir, name)
        try:
            self.logger.write_videos(
                step=step, images=rendering_imgs, logdir=path_render_path
            )
        except:
            print("Video logging error. Likely a system problem.")

    def save_model(self, e):
        # save checkpoint
        name = f"extractor_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)
        model = deepcopy(self.extractor).to("cpu")
        torch.save(model.state_dict(), path)
