import os
import random
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.layers.base import Base
from trainer.base_trainer import BaseTrainer
from utils.sampler import OnlineSampler


# model-free policy trainer
class OnPolicyTrainer(BaseTrainer):
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

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        wall_clock_time = 0
        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                batch, sample_time = self.sampler.collect_samples(
                    env=self.env, policy=self.policy, seed=self.seed
                )
                loss_dict, timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(timesteps)
                wall_clock_time += sample_time + update_time

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/timesteps"] = step + timesteps
                loss_dict[f"{self.policy.name}/analytics/wall_clock_time (hr)"] = (
                    wall_clock_time / 3600.0
                )
                loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time

                self.write_log(loss_dict, step=step)

                #### EVALUATIONS ####
                if step >= self.eval_interval * (eval_idx + 1):
                    ### Eval Loop
                    self.policy.eval()
                    eval_idx += 1

                    eval_dict, supp_dict = self.evaluate()

                    # Manual logging
                    if self.policy.state_visitation is not None:
                        visitation_map = self.policy.state_visitation
                        vmin, vmax = visitation_map.min(), visitation_map.max()
                        visitation_map = (visitation_map - vmin) / (vmax - vmin + 1e-8)
                        visitation_map = self.visitation_to_rgb(visitation_map)
                        self.write_image(
                            image=visitation_map,
                            step=step,
                            logdir="Image",
                            name="visitation map",
                        )

                    self.write_log(eval_dict, step=step, eval_log=True)
                    self.write_image(
                        supp_dict["eval/trajectory_plot"],
                        step=step,
                        logdir="Image",
                        name="trajectory",
                    )
                    self.write_video(
                        supp_dict["eval/rendering"],
                        step=step,
                        logdir=f"Video",
                        name="running_video",
                    )

                    self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                    self.last_return_std.append(eval_dict[f"eval/return_std"])

                    self.save_model(step)

                torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

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
                np.mean(self.last_return_mean) >= self.last_max_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_max_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
