import glob
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils.functions import call_env
from utils.sampler import OnlineSampler


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, logger, writer, args):
        super(IntrinsicRewardFunctions, self).__init__()

        # === Parameter saving === #
        self.num_trials = 2_000

        self.extractor_env = call_env(deepcopy(args), random_spawn=True)
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        # === MAKE ENV === #
        if self.args.intrinsic_reward_mode == "allo":
            self.num_rewards = self.args.num_options
            self.extractor_mode = "allo"
            self.define_extractor()
            self.define_eigenvectors()
            self.define_intrinsic_reward_normalizer()

            self.sources = ["allo" for _ in range(self.num_rewards)]
        elif self.args.intrinsic_reward_mode == "random":
            self.num_rewards = self.args.num_options
            self.extractor_mode = "random"

            self.define_random_model()
            self.define_eigenvectors()
            self.define_intrinsic_reward_normalizer()

            self.sources = ["random" for _ in range(self.args.num_options)]
        else:
            raise NotImplementedError(
                f"The intrinsic reward mode {self.args.intrinsic_reward_mode} not implemented or unknown."
            )

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, i: int):
        if self.sources[i] == "allo":
            states = states[:, self.args.positional_indices]
            next_states = next_states[:, self.args.positional_indices]

            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature

                eigenvector_idx, eigenvector_sign = self.eigenvectors[i]
                intrinsic_rewards = eigenvector_sign * difference[
                    :, eigenvector_idx
                ].unsqueeze(-1)
        elif self.sources[i] == "random":
            states = states[:, self.args.positional_indices]
            next_states = next_states[:, self.args.positional_indices]

            with torch.no_grad():
                feature, _ = self.extractor(states)
                next_feature, _ = self.extractor(next_states)
                difference = next_feature - feature

                eigenvector_idx, eigenvector_sign = self.eigenvectors[i]
                intrinsic_rewards = eigenvector_sign * difference[
                    :, eigenvector_idx
                ].unsqueeze(-1)

        if hasattr(self, "reward_rms"):
            self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())
            var_tensor = torch.as_tensor(
                self.reward_rms[i].var,
                device=intrinsic_rewards.device,
                dtype=intrinsic_rewards.dtype,
            )
            intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

        return intrinsic_rewards, self.sources[i]

    def learn(
        self, states: torch.Tensor, next_states: torch.Tensor, i: int, keyword: str
    ):
        pass

    def define_extractor(self):
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO
        from policy.uniform_random import UniformRandom
        from trainer.extractor_trainer import ExtractorTrainer

        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(f"model/{self.args.env_name}"):
            os.makedirs(f"model/{self.args.env_name}")

        # === CREATE FEATURE EXTRACTOR === #
        feature_network = NeuralNet(
            state_dim=len(self.args.positional_indices),
            feature_dim=self.args.feature_dim,
            encoder_fc_dim=[512, 512, 512, 512],
            activation=nn.LeakyReLU(),
        )

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        extractor = ALLO(
            network=feature_network,
            positional_indices=self.args.positional_indices,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.extractor_epochs,
            batch_size=1024,
            lr_barrier_coeff=self.args.lr_barrier_coeff,  # ALLO uses 0.01 lr_barrier_coeff
            discount=self.args.discount_sampling_factor,  # ALLO uses 0.99 discount
            device=self.args.device,
        )

        # Step 1: Search for .pth files in the directory
        model_dir = f"model/{self.args.env_name}/"
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if not pth_files:
            print(
                f"[INFO] No existing model found in {model_dir}. Training from scratch."
            )
            epochs = 0
            model_path = os.path.join(
                model_dir,
                f"ALLO_{self.args.extractor_epochs}_{self.args.discount_sampling_factor}.pth",
            )
        else:
            print(f"[INFO] Found {len(pth_files)} .pth files in {model_dir}")
            epochs = []
            discount_factors = []
            valid_files = []

            for pth_file in pth_files:
                filename = os.path.basename(pth_file)
                parts = filename.replace(".pth", "").split("_")
                if len(parts) != 3:
                    print(f"[WARNING] Skipping malformed file: {filename}")
                    continue

                _, epoch_str, discount_str = parts
                try:
                    epoch = int(epoch_str)
                    discount = float(discount_str)
                    epochs.append(epoch)
                    discount_factors.append(discount)
                    valid_files.append(filename)
                except ValueError:
                    print(f"[WARNING] Failed to parse file: {filename}")
                    continue

            if self.args.discount_sampling_factor not in discount_factors:
                print(
                    f"[INFO] No model with discount factor {self.args.discount_sampling_factor} found. Starting fresh."
                )
                epochs = 0
                model_path = os.path.join(
                    model_dir,
                    f"ALLO_{self.args.extractor_epochs}_{self.args.discount_sampling_factor}.pth",
                )
            else:
                matching = [
                    (e, f, filename)
                    for e, f, filename in zip(epochs, discount_factors, valid_files)
                    if f == self.args.discount_sampling_factor
                ]

                max_epoch, _, _ = max(matching, key=lambda x: x[0])
                idx = epochs.index(max_epoch)
                filename = matching[idx][-1]
                model_path = os.path.join(model_dir, filename)
                print(
                    f"[INFO] Loading model from: {model_path} (epoch {max_epoch}, discount {self.args.discount_sampling_factor})"
                )

                extractor.load_state_dict(
                    torch.load(model_path, map_location=self.args.device)
                )
                extractor.to(self.args.device)
                epochs = max_epoch  # set current epoch

        if epochs < self.args.extractor_epochs:
            uniform_random_policy = UniformRandom(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                is_discrete=self.args.is_discrete,
                device=self.args.device,
            )
            sampler = OnlineSampler(
                state_dim=self.args.state_dim,
                action_dim=self.args.action_dim,
                episode_len=self.args.episode_len,
                batch_size=self.num_trials * self.args.episode_len,
                verbose=False,
            )
            trainer = ExtractorTrainer(
                env=self.extractor_env,
                random_policy=uniform_random_policy,
                extractor=extractor,
                sampler=sampler,
                logger=self.logger,
                writer=self.writer,
                epochs=self.args.extractor_epochs - epochs,
            )
            final_timesteps = trainer.train()
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)

        self.extractor = extractor

    def define_eigenvectors(self):
        # === Define eigenvectors === #
        # ALLO does not have explicit eigenvectors.
        # Instead, we make list that contains the eigenvector index and sign
        eigenvectors = [
            (n // 2 + 1, 2 * (n % 2) - 1) for n in range(self.args.num_options)
        ]

        heatmaps = self.extractor_env.get_rewards_heatmap(self.extractor, eigenvectors)
        log_dir = f"{self.logger.log_dir}/intrinsic_rewards"

        os.mkdir(f"{log_dir}")
        for i, fig in enumerate(heatmaps):
            if isinstance(fig, np.ndarray):
                plt.imsave(f"{log_dir}/figure_{i}.pdf", fig, cmap="viridis")
                plt.imsave(f"{log_dir}/figure_{i}.svg", fig, cmap="viridis")
            elif isinstance(fig, plt.Figure):
                fig.savefig(f"{log_dir}/figure_{i}.pdf", format="pdf")
                fig.savefig(f"{log_dir}/figure_{i}.svg", format="svg")

        self.eigenvectors = eigenvectors
        self.logger.write_images(
            step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
        )

    def define_random_model(self):
        # random arbitrary neural networks size of (32, 32)
        # but with different nonlinearities and initializations
        from extractor.base.mlp import NeuralNet
        from extractor.extractor import ALLO

        # === CREATE FEATURE EXTRACTOR === #
        feature_network = NeuralNet(
            state_dim=len(self.args.positional_indices),
            feature_dim=self.args.feature_dim,
            encoder_fc_dim=[512, 512, 512, 512],
            activation=nn.Tanh(),
        )

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        self.extractor = ALLO(
            network=feature_network,
            positional_indices=self.args.positional_indices,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.extractor_epochs,
            batch_size=1024,
            lr_barrier_coeff=self.args.lr_barrier_coeff,  # ALLO uses 0.01 lr_barrier_coeff
            discount=self.args.discount_sampling_factor,  # ALLO uses 0.99 discount
            device=self.args.device,
        )

    def define_intrinsic_reward_normalizer(self):
        from utils.wrapper import RunningMeanStd

        self.reward_rms = []
        for _ in range(self.args.num_options):
            self.reward_rms.append(RunningMeanStd(shape=(1,)))
