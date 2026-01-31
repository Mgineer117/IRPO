import random
from itertools import chain
from typing import (
    Any,
    Final,
    Iterable,
    Literal,
    SupportsFloat,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray

from gridworld.core.agent import Agent, AgentT, GridActions, PolicyAgent
from gridworld.core.constants import *
from gridworld.core.grid import Grid
from gridworld.core.object import Goal, Lava, Obstacle, Wall
from gridworld.core.world import GridWorld
from gridworld.multigrid import MultiGridEnv
from gridworld.policy.ctf.heuristic import (
    HEURISTIC_POLICIES,
    CtfPolicyT,
    RoombaPolicy,
    RwPolicy,
)
from gridworld.typing import Position
from gridworld.utils.window import Window


class Maze(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: int,
        max_steps: int,
        num_random_agent: int = 0,
        highlight_visible_cells: bool = False,
        tile_size: int = 10,
        state_representation: str = "positional",
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        self.state_representation = state_representation
        if grid_type < 0 or grid_type >= 2:
            raise ValueError(
                f"The Fourroom only accepts grid_type of 0 and 1, given {grid_type}"
            )
        else:
            self.grid_type = grid_type

        self.max_steps = max_steps

        self.world = GridWorld
        self.actions_set = GridActions

        see_through_walls: bool = False

        self.agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                actions=self.actions_set,
                type="agent",
            )
        ]

        # Define positions for goals and agents
        self.goal_positions = [(7, 3), (1, 9), (1, 5)]
        self.agent_positions = [(5, 9), (9, 5), (3, 5)]

        self.num_random_agent = num_random_agent
        self.random_agent_positions = [
            [(6, 5), (6, 14)],
        ]
        for i in range(1, 1 + self.num_random_agent):
            self.agents.append(
                PolicyAgent(
                    RwPolicy(action_set=self.actions_set),
                    self.world,
                    index=i,
                    color="red",
                    bg_color="light_red",
                    actions=self.actions_set,
                    type="obstacle",
                )
            )

        self.grids = {}
        self.grid_imgs = {}
        # Explicit maze structure based on the image
        self.map_structure = [
            [
                "###########",
                "#     #   #",
                "# ### # # #",
                "#   # # # #",
                "### # ### #",
                "# # # #   #",
                "# # # # ###",
                "# # # #   #",
                "# ### ### #",
                "#         #",
                "###########",
            ],
            [
                "###########",
                "#         #",
                "# ####### #",
                "#       # #",
                "# # ##### #",
                "# #   #   #",
                "# # # # # #",
                "# # #   # #",
                "# # ##### #",
                "# #       #",
                "###########",
            ],
            [
                "###########",
                "#   #     #",
                "# # # ### #",
                "# #   #   #",
                "# ### # ###",
                "# # # #   #",
                "### # ### #",
                "#   # #   #",
                "# ##### # #",
                "#       # #",
                "###########",
            ],
        ]

        self.width = len(self.map_structure[grid_type][0])
        self.height = len(self.map_structure[grid_type])
        self.grid_size = (self.width, self.height)

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def get_grid(self):
        self.reset()
        grid = self.grid.encode()
        self.close()
        return grid

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.state_representation:
            case "positional":
                observation_space = spaces.Box(
                    low=np.array([0, 0, 0, 0], dtype=np.float32),
                    high=np.array(
                        [self.width, self.height, self.width, self.height],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=13,
                    shape=(self.width, self.height, self.world.encode_dim),
                    dtype=np.int64,
                )
            case "vectorized_tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=10,
                    shape=(self.width * self.height * self.world.encode_dim,),
                    dtype=np.int64,
                )
            case _:
                raise ValueError(
                    f"Invalid state representation: {self.state_representation}"
                )

        return observation_space

    def _gen_grid(self, width, height, options):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Translate the maze structure into the grid
        for y, row in enumerate(self.map_structure[self.grid_type]):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                elif cell == " ":
                    self.grid.set(x, y, None)

        # Place the goal
        goal = Goal(self.world, index=4)
        self.put_obj(goal, *self.goal_positions[self.grid_type])
        goal.init_pos, goal.cur_pos = self.goal_positions[self.grid_type]

        # place agent
        if options.get("random_init_pos"):
            coords = self.find_obj_coordinates(None)
            agent_positions = random.sample(coords, 1)[0]
        else:
            agent_positions = self.agent_positions[self.grid_type]
        self.place_agent(self.agents[0], pos=agent_positions)

        if len(self.agents) > 1:
            for i, agent in enumerate(self.agents[1:]):
                self.place_agent(
                    agent, pos=self.random_agent_positions[self.grid_type][i]
                )

    def find_obj_coordinates(self, obj) -> tuple[int, int] | None:
        """
        Finds the coordinates (i, j) of the first occurrence of None in the grid.
        Returns None if no None value is found.
        """
        coord_list = []
        for index, value in enumerate(self.grid.grid):
            if value is obj:
                # Calculate the (i, j) coordinates from the 1D index
                i = index % self.width
                j = index // self.width
                coord_list.append((i, j))
        return coord_list

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict = {},
    ):
        super().reset(seed=seed, options=options)

        ### NOTE: NOT MULTIAGENT SETTING
        observations = self.get_obs()
        info = {"success": False}

        return observations, info

    def step(self, action):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        action = np.argmax(action)
        actions = [action]

        for i in range(1, len(self.agents)):
            actions.append(self.agents[i].policy.act())

        rewards = np.zeros(1)
        info = {"success": False}
        done = False

        for i in range(len(self.agents)):
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
            ):
                continue

            # Get the current agent position
            curr_pos = self.agents[i].pos

            # Rotate left
            if actions[i] == self.actions.left:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, -1)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if self.agents[i].type == "agent":
                        if fwd_cell.type == "goal":
                            done = True
                            rewards = self._reward(i, rewards, 1)
                            info["success"] = True
                        elif fwd_cell.type == "lava":
                            done = True
                            rewards[i] = -1
                            info["success"] = False
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if self.agents[i].type == "agent":
                        if fwd_cell.type == "goal":
                            done = True
                            rewards = self._reward(i, rewards, 1)
                            info["success"] = True
                        elif fwd_cell.type == "lava":
                            done = True
                            rewards[i] = -1
                            info["success"] = False
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if self.agents[i].type == "agent":
                        if fwd_cell.type == "goal":
                            done = True
                            rewards = self._reward(i, rewards, 1)
                            info["success"] = True
                        elif fwd_cell.type == "lava":
                            done = True
                            rewards[i] = -1
                            info["success"] = False
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is not None:
                    if self.agents[i].type == "agent":
                        if fwd_cell.type == "goal":
                            done = True
                            rewards = self._reward(i, rewards, 1)
                            info["success"] = True
                        elif fwd_cell.type == "lava":
                            done = True
                            rewards[i] = -1
                            info["success"] = False
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            elif actions[i] == self.actions.stay:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
                self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
        terminated = done
        truncated = True if self.step_count >= self.max_steps else False

        observations = self.get_obs()

        return observations, rewards, terminated, truncated, info

    def _reward(self, current_agent, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        rewards[current_agent] += reward
        return rewards

    def get_obs(
        self,
    ):
        if self.state_representation == "positional":
            obs = {
                "achieved_goal": np.array(
                    [self.agents[0].pos[0], self.agents[0].pos[1]]
                ),
                "desired_goal": np.array(
                    [
                        self.goal_positions[self.grid_type][0],
                        self.goal_positions[self.grid_type][1],
                    ]
                ),
            }
        elif self.state_representation == "tensor":
            obs = self.grid.encode()
        elif self.state_representation == "vectorized_tensor":
            obs = self.grid.encode().flatten()
        else:
            raise ValueError(
                f"Unknown state representation {self.state_representation}. "
                "Please use 'positional' or 'tensor'."
            )
        return obs

    def get_rewards_heatmap(
        self, extractor: torch.nn.Module, eigenvectors: np.ndarray | list
    ):
        assert self.state_representation in [
            "vectorized_tensor",
            "tensor",
            "positional",
        ], f"Unsupported state representation: {self.state_representation}"

        # Environment indices
        empty_idx = 1
        goal_idx = 8
        agent_idx = 10
        obs_idx = 13
        wall_idx = 2

        # Get base state
        state = self.get_grid()
        agent_pos = np.where(state == agent_idx)
        state[agent_pos] = empty_idx
        grid = state

        mask = (grid != wall_idx) & (grid != goal_idx)

        # Get coordinates where agent can be placed
        valid_coords = np.argwhere(mask)  # shape: [num_valid, 2]

        # Generate a batch of states
        state_batch = []
        for coord in valid_coords:
            if self.state_representation in ("tensor", "vectorized_tensor"):
                new_grid = grid.copy()
                new_grid[new_grid[..., 0] == agent_idx] = empty_idx
                new_grid[coord[0], coord[1], 0] = agent_idx

                state_batch.append(new_grid)
            elif self.state_representation == "positional":
                state = np.array([coord[0], coord[1]])
                state_batch.append(state)

        # Stack the batch: shape = [num_valid, H, W] or [num_valid, H, W, C]
        state_batch = np.stack(state_batch)

        heatmaps = []
        grid_shape = (self.width, self.height, 1)
        for n in range(len(eigenvectors)):
            reward_map = np.full(grid_shape, fill_value=0.0)

            with torch.no_grad():
                features, _ = extractor(state_batch)
                features = features.cpu().numpy()

            for i in range(features.shape[0]):
                if self.state_representation in ("tensor", "vectorized_tensor"):
                    agent_pos = np.argwhere(state_batch[i] == agent_idx)[0]
                elif self.state_representation == "positional":
                    agent_pos = [state_batch[i][0], state_batch[i][1]]
                x, y = agent_pos[0], agent_pos[1]

                eigenvector_idx, eigenvector_sign = eigenvectors[n]
                reward = eigenvector_sign * features[i, eigenvector_idx]

                reward_map[x, y, 0] = reward

            # reward_map = # normalize between -1 to 1
            pos_mask = np.logical_and(mask, (reward_map > 0))
            neg_mask = np.logical_and(mask, (reward_map < 0))

            # Normalize positive values to [0, 1]
            if np.any(pos_mask):
                pos_max, pos_min = (
                    reward_map[pos_mask].max(),
                    reward_map[pos_mask].min(),
                )
                if pos_max != pos_min:
                    reward_map[pos_mask] = (reward_map[pos_mask] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_mask):
                neg_max, neg_min = (
                    reward_map[neg_mask].max(),
                    reward_map[neg_mask].min(),
                )
                if neg_max != neg_min:
                    reward_map[neg_mask] = (reward_map[neg_mask] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Set all other entries (walls, empty) to 0
            # print(reward_map[:, :, 0])
            reward_map = self.reward_map_to_rgb(reward_map, mask)

            # set color theme as blue and red (blue = -1 and red = 1)
            # set wall color at value 0 and goal idx as 1
            heatmaps.append(reward_map)

        return heatmaps

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map > 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask[:, :, 0], 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask[:, :, 0], 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask[:, :, 0], :] = 0.5

        return rgb_img
