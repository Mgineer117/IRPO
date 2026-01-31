import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        tot_count = self.count + count

        new_mean = self.mean + delta * count / tot_count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + np.square(delta) * self.count * count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class GridWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(GridWrapper, self).__init__(env)

    def reset(self, **kwargs):
        observation_dict, info = self.env.reset(**kwargs)
        info.update(observation_dict)

        observation = np.concatenate(
            (
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        return observation, info

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        info.update(observation_dict)

        observation = np.concatenate(
            (
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )
        return observation, reward, termination, truncation, info

    def get_trajectory_info(self):
        return None, None

    def get_trajectory_plot(self, trajectories: list, desired_goals: list):
        return None

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)


class FetchWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, episode_len: int, seed: int):
        super(FetchWrapper, self).__init__(env)

        self.max_steps = episode_len
        self.seed = seed

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation_dict, info = self.env.reset(**kwargs)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        return observation, info

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        reward += 1.0  # to scale reawrd [0, 1]

        # if reward == 1.0:
        if reward == 1.0:
            termination = True  # terminate if goal is achieved

        return observation, reward, termination, truncation, info

    def get_trajectory_info(self):
        return None, None

    def get_trajectory_plot(self, trajectories: list, desired_goals: list):
        return None

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors: np.ndarray):

        state, _ = self.reset(seed=self.seed)
        dg = state[-6:-3]  # desired goal pos
        del state
        self.close()

        X = [0.5, 1.5]
        Y = [0, 1.0]
        Z = [0.5, 1.0]

        # Define the ranges and number of increments per dimension
        x_vals = np.linspace(X[0], X[1], num=10)
        y_vals = np.linspace(Y[0], Y[1], num=10)
        z_vals = np.linspace(Z[0], Z[1], num=10)

        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Flatten to create a batch of shape (N, 3)
        states = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

        with torch.no_grad():
            intrinsic_rewards, _ = extractor(states)
            intrinsic_rewards = intrinsic_rewards.cpu().numpy()

        images = []
        for eigenvector_idx, eigenvector_sign in eigenvectors:
            # Compute reward as dot product
            # rewards = achieved_goals @ vector  # shape: (N,)
            rewards = eigenvector_sign * intrinsic_rewards[:, eigenvector_idx]

            neg_idx = rewards < 0
            pos_idx = rewards >= 0

            # Normalize positive values to [0, 1]
            if np.any(pos_idx):
                pos_max, pos_min = (
                    rewards[pos_idx].max(),
                    rewards[pos_idx].min(),
                )
                if pos_max != pos_min:
                    rewards[pos_idx] = (rewards[pos_idx] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_idx):
                neg_max, neg_min = (
                    rewards[neg_idx].max(),
                    rewards[neg_idx].min(),
                )
                if neg_max != neg_min:
                    rewards[neg_idx] = (rewards[neg_idx] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Create 3D scatter plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                c=rewards,
                cmap=cm.seismic,
                s=30,
            )

            # Mark desired goal
            ax.scatter(
                dg[0],
                dg[1],
                dg[2],
                color="yellow",
                edgecolors="black",
                s=500,
                marker="*",
                label="Desired Goal",
            )

            # ax.set_title(
            #     f"Rewards for Eigenvector {eigenvector_idx}-{eigenvector_sign}"
            # )
            ax.set_xlabel("X", fontsize=18)
            ax.set_ylabel("Y", fontsize=18)
            ax.set_zlabel("Z", fontsize=18)
            ax.legend(fontsize=14)

            # plt.colorbar(sc, ax=ax, label="Normalized Reward")

            images.append(fig)
            plt.close()

            # Convert plot to image
            # buf = BytesIO()
            # plt.tight_layout()
            # plt.savefig(buf, format="png")
            # plt.close(fig)
            # buf.seek(0)
            # image = Image.open(buf).convert("RGB")
            # images.append(np.array(image))
            # buf.close()

        return images


class MazeWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        maze_map: list,
        episode_len: int,
        seed: int,
        cell_size: int = 1.0,
    ):
        super(MazeWrapper, self).__init__(env)

        self.maze_map = maze_map
        self.max_steps = episode_len
        self.seed = seed

        self.trajectory = []
        self.desired_goal = None

        self.cell_size = cell_size

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation_dict, info = self.env.reset(**kwargs)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        self.desired_goal = observation_dict["desired_goal"].copy()
        self.trajectory = [observation_dict["achieved_goal"].copy()]

        return observation, info

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        self.trajectory.append(observation_dict["achieved_goal"].copy())
        return observation, reward, termination, truncation, info

    def get_trajectory_info(self):
        trajectory = self.trajectory
        self.trajectory = None
        return trajectory, self.desired_goal

    def get_trajectory_plot(self, trajectories: list, desired_goals: list):
        fig = plt.figure(figsize=(8, 6))

        # Maze info
        maze_height = len(self.maze_map)
        maze_width = len(self.maze_map[0])
        cell_size = self.cell_size

        # Draw walls
        for i in range(maze_height):
            for j in range(maze_width):
                val = self.maze_map[i][j]
                if val == 1:
                    # Compute cell center
                    x_center = (j + 0.5) * cell_size - (maze_width * cell_size / 2)
                    y_center = (maze_height * cell_size / 2) - (i + 0.5) * cell_size
                    # Draw rectangle
                    rect = plt.Rectangle(
                        (x_center - cell_size / 2, y_center - cell_size / 2),
                        cell_size,
                        cell_size,
                        color="black",
                    )
                    plt.gca().add_patch(rect)

        # Plot trajectory
        for i, traj in enumerate(trajectories):
            traj = np.array(traj)
            # mark start and end points
            plt.scatter(
                traj[0, 0],
                traj[0, 1],
                color="green",
                label="Start" if i == 0 else None,
                s=50,
            )
            plt.scatter(
                traj[-1, 0],
                traj[-1, 1],
                color="red",
                label="End" if i == 0 else None,
                s=50,
            )
            plt.plot(
                traj[:, 0],
                traj[:, 1],
                color=plt.cm.viridis(i / len(trajectories)),
                label=f"Trajectory" if i == 0 else None,
                lw=2,
                alpha=0.8,
            )

            # Plot desired goal with star marker with boundary black
            plt.scatter(
                desired_goals[i][0],
                desired_goals[i][1],
                color=plt.cm.viridis(i / len(trajectories)),
                label="Goal" if i == 0 else None,
                s=300,
                marker="*",
                edgecolor="black",
            )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Agent Trajectory in Maze")
        plt.legend(fontsize=14)
        plt.axis("equal")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.close()

        return fig

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors: np.ndarray):
        # Get desired goal (2D)
        state, _ = self.reset(seed=self.seed)
        dg = state[-4:-2]
        del state
        self.close()

        # Maze size
        example_map = self.maze_map
        maze_height = len(example_map)
        maze_width = len(example_map[0])

        # Spatial bounds (MuJoCo centered coordinates)
        cell_size = 1.0
        x_low = -maze_width * cell_size / 2
        x_high = maze_width * cell_size / 2
        y_low = -maze_height * cell_size / 2
        y_high = maze_height * cell_size / 2
        resolution = 80

        x_vals = np.linspace(x_low, x_high, num=resolution)
        y_vals = np.linspace(y_low, y_high, num=resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
        states = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

        self.width = resolution
        self.height = resolution

        # Create wall/goal/agent masks
        wall_mask = np.zeros((resolution, resolution), dtype=bool)
        goal_mask = np.zeros((resolution, resolution), dtype=bool)
        agent_mask = np.zeros((resolution, resolution), dtype=bool)

        for i in range(maze_height):
            for j in range(maze_width):
                val = example_map[i][j]

                # MuJoCo coordinate of cell center
                x_center = (j + 0.5) * cell_size - (maze_width * cell_size / 2)
                y_center = (maze_height * cell_size / 2) - (i + 0.5) * cell_size

                x_start = x_center - cell_size / 2
                x_end = x_center + cell_size / 2
                y_start = y_center - cell_size / 2
                y_end = y_center + cell_size / 2

                region_mask = (
                    (X_grid >= x_start)
                    & (X_grid < x_end)
                    & (Y_grid >= y_start)
                    & (Y_grid < y_end)
                )

                if val == 1:
                    wall_mask |= region_mask
                elif val == "g":
                    goal_mask |= region_mask
                elif val == "r":
                    agent_mask |= region_mask

        valid_mask = ~wall_mask

        # Loop over each eigenvector to generate heatmaps
        images = []
        with torch.no_grad():
            intrinsic_rewards, _ = extractor(states)
        intrinsic_rewards = intrinsic_rewards.cpu().numpy()

        for eigenvector_idx, eigenvector_sign in eigenvectors:
            rewards = eigenvector_sign * intrinsic_rewards[:, eigenvector_idx]

            # Normalize rewards separately for positive and negative
            neg_idx = rewards < 0
            pos_idx = rewards >= 0

            if np.any(pos_idx):
                pos_max, pos_min = rewards[pos_idx].max(), rewards[pos_idx].min()
                if pos_max != pos_min:
                    rewards[pos_idx] = (rewards[pos_idx] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )
                else:
                    rewards[pos_idx] = 1
            if np.any(neg_idx):
                neg_max, neg_min = rewards[neg_idx].max(), rewards[neg_idx].min()
                if neg_max != neg_min:
                    rewards[neg_idx] = (rewards[neg_idx] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0
                else:
                    rewards[neg_idx] = -1

            reward_map = rewards.reshape(resolution, resolution)
            rgb_img = self.reward_map_to_rgb(reward_map, valid_mask)
            images.append(rgb_img)

        return images

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map >= 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask, 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask, 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask, :] = 0.5

        return rgb_img


class AantMazeWrapper(MazeWrapper):
    def __init__(
        self,
        env: gym.Env,
        maze_map: list,
        episode_len: int,
        seed: int,
        cell_size: int = 1.0,
    ):
        super().__init__(
            env,
            maze_map=maze_map,
            episode_len=episode_len,
            seed=seed,
            cell_size=cell_size,
        )

        self.healthy_penalty = 1.0

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["achieved_goal"],
                observation_dict["desired_goal"],
            )
        )

        if (
            observation_dict["observation"][0] >= 1.00
            or observation_dict["observation"][0] <= 0.35
        ):
            reward -= self.healthy_penalty
            termination = True

        self.trajectory.append(observation_dict["achieved_goal"].copy())
        return observation, reward, termination, truncation, info
