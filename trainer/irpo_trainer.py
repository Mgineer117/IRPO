import os
import random
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from trainer.base_trainer import BaseTrainer
from utils.sampler import OnlineSampler


# model-free policy trainer
class IRPOTrainer(BaseTrainer):
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
        args=None,
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
        self.last_min_return_mean = 1e10
        self.last_min_return_std = 1e10

        self.rendering = rendering
        self.seed = seed
        self.args = args

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0

        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                current_step = pbar.n

                loss_dict, timesteps, visitation_dict = self.policy.learn(
                    self.env,
                    self.sampler,
                    self.seed,
                    current_step / (self.timesteps + self.init_timesteps),
                )
                current_step = pbar.n + timesteps

                # === Update progress === #
                self.write_log(loss_dict, current_step)
                pbar.update(timesteps)

                # === EVALUATIONS === #
                if current_step >= self.eval_interval * (eval_idx + 1):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, supp_dict = self.evaluate()

                    # Manual logging
                    weights = self.policy.probability_history
                    weights = (weights - weights.min()) / (
                        weights.max() - weights.min() + 1e-8
                    )
                    weights = weights / (weights.sum() + 1e-8)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.stem(
                        np.array([int(x) for x in self.policy.contributing_indices]),
                        weights,
                    )

                    # Convert figure to a NumPy array
                    canvas = FigureCanvas(fig)
                    canvas.draw()

                    # Use buffer_rgba instead of tostring_rgb
                    img = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
                    img = img.reshape(
                        canvas.get_width_height()[::-1] + (4,)
                    )  # (H, W, 4)

                    # Drop the alpha channel to get RGB
                    img = img[:, :, :3]  # Shape: (H, W, 3)

                    plt.close()

                    if visitation_dict is not None:
                        for key, value in visitation_dict.items():
                            visitation_map = value
                            vmin, vmax = visitation_map.min(), visitation_map.max()
                            visitation_map = (visitation_map - vmin) / (
                                vmax - vmin + 1e-8
                            )
                            visitation_map = self.visitation_to_rgb(visitation_map)
                            self.write_image(
                                image=visitation_map,
                                step=current_step,
                                logdir=f"Image",
                                name=key,
                            )

                    self.write_log(eval_dict, step=current_step, eval_log=True)
                    self.write_image(
                        supp_dict["eval/trajectory_plot"],
                        step=current_step,
                        logdir="Image",
                        name="trajectory",
                    )
                    self.write_video(
                        supp_dict["eval/rendering"],
                        step=current_step,
                        logdir=f"Video",
                        name="running_video",
                    )
                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(current_step)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

        return current_step

    def evaluate(self):
        ep_buffer, image_array, trajectories, desired_goals = [], [], [], []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            eval_seed = random.randint(0, 10000) + self.seed + num_episodes
            state, infos = self.env.reset(seed=eval_seed)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    a, _ = self.policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    # Plotting
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(a)
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(
                                ep_reward, self.policy.gamma
                            ),
                        }
                    )

                    traj, desired_goal = self.env.get_trajectory_info()
                    trajectories.append(traj)
                    desired_goals.append(desired_goal)
                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)

        spectral_loss = self.spectral_loss(self.policy.actor)
        trajectory_plot = self.env.get_trajectory_plot(trajectories, desired_goals)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
            f"eval/spectral_loss": spectral_loss,
        }
        supp_dict = {
            "eval/trajectory_plot": trajectory_plot,
            "eval/rendering": image_array,
        }

        return eval_dict, supp_dict

    def save_model(self, e):
        ### save checkpoint
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = self.policy.actor

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) < self.last_min_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_min_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
