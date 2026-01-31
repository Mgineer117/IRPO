import random
from typing import Any, Final, Literal, Type, TypeAlias, TypedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gridworld.core.agent import Agent, AgentT, GridActions, PolicyAgent
from gridworld.core.grid import Grid
from gridworld.core.object import Flag, Floor, Obstacle, WorldObjT
from gridworld.core.world import CtfWorld
from gridworld.multigrid import MultiGridEnv
from gridworld.policy.ctf.heuristic import (
    HEURISTIC_POLICIES,
    CtfPolicyT,
    RoombaPolicy,
    RwPolicy,
)
from gridworld.policy.ctf.typing import ObservationDict
from gridworld.typing import Position
from gridworld.utils.map import distance_area_point, distance_points, load_text_map

Observation: TypeAlias = (
    ObservationDict | NDArray[np.int_] | dict[str, NDArray[np.int_] | int]
)


class GameStats(TypedDict):
    defeated_blue_agents: int
    defeated_red_agents: int
    captured_blue_flags: int
    captured_red_flags: int
    blue_agent_defeated: list[bool]
    red_agent_defeated: list[bool]
    blue_flag_captured: bool
    red_flag_captured: bool


ObservationOption: TypeAlias = Literal[
    "positional", "map", "flattened", "pos_map", "pos_map_flattened", "tensor"
]


class CtfMvNEnv(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: int,
        enemy_policies: (
            list[Type[CtfPolicyT]] | Type[CtfPolicyT] | list[str] | str
        ) = RoombaPolicy(),
        enemy_policy_kwargs: list[dict[str, Any]] | dict[str, Any] = {},
        battle_range: float = 1,
        territory_adv_rate: float = 0.75,
        flag_reward: float = 1,
        battle_reward_ratio: float = 0.25,
        obstacle_penalty_ratio: float = 0,
        step_penalty_ratio: float = 0.01,
        max_steps: int = 100,
        observation_option: ObservationOption = "positional",
        observation_scaling: float = 1,
        render_mode: Literal["human"] | Literal["rgb_array"] = "rgb_array",
        uncached_object_types: list[str] = ["red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize a new capture the flag environment.

        Parameters
        ----------
        map_path : str
            Path to the map file.
        num_blue_agents : int = 2
            Number of blue (friendly) agents.
        num_red_agents : int = 2
            Number of red (enemy) agents.
        enemy_policies : list[CtfPolicyT] | CtfPolicyT | list[str] | str = RwPolicy()
            Policies of the enemy agents.
            If there is only one policy, it will be used for all enemy agents.
            If there is a list of policies, the number of policies should be equal to the number of enemy agents.
            If the policy is a string, it should be one of the keys in HEURISTIC_POLICIES.
        enemy_policy_kwargs : list[dict[str, Any]] | dict[str, Any] = {"avoided_objects": ["obstacle", "blue_agent", "red_agent"]}
            Configuration for the enemy policies.
            If there is only one configuration, it will be used for all enemy policies.
            If there is a list of configurations, the number of configurations should be equal to the number of enemy agents.
        battle_range : float = 1
            Range within which battles can occur.
        territory_adv_rate : float = 0.75
            Probability of the enemy agent winning a battle within its territory.
        flag_reward : float = 1
            Reward for capturing the enemy flag.
        battle_reward_ratio : float = 0.25
            Ratio of the flag reward for winning a battle.
        obstacle_penalty_ratio : float=0
            Ratio of the flag reward for colliding with an obstacle.
        step_penalty_ratio : float = 0.01
            Ratio of the flag reward for taking a step.
        max_steps : int=100
            Maximum number of steps per episode.
        observation_option : Literal["positional", "map", "flattened"] = "positional"
            Observation option.
        observation_scaling : float = 1
            Scaling factor for the observation.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Rendering mode.
        uncached_object_types : list[str] = ["red_agent", "blue_agent"]
            Types of objects that should not be cached.
        """

        assert battle_range > 0
        assert 0 <= territory_adv_rate <= 1
        assert flag_reward > 0
        assert 0 <= battle_reward_ratio <= 1
        assert 0 <= obstacle_penalty_ratio <= 1
        assert 0 <= step_penalty_ratio <= 1
        assert max_steps > 0
        assert observation_scaling > 0

        self.battle_range: Final[float] = battle_range
        self.randomness: Final[float] = territory_adv_rate
        self.flag_reward: Final[float] = flag_reward
        self.battle_reward: Final[float] = battle_reward_ratio * flag_reward
        self.obstacle_penalty: Final[float] = obstacle_penalty_ratio * flag_reward
        self.step_penalty: Final[float] = step_penalty_ratio * flag_reward

        self.observation_option: Final[ObservationOption] = observation_option
        self.observation_scaling: Final[float] = observation_scaling

        partial_obs: bool = False
        agent_view_size: int = 10

        self.world = CtfWorld
        self.actions_set = GridActions
        see_through_walls: bool = False

        if grid_type == 0:
            ctf_grid = np.array(
                [
                    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                    [6, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 6],
                    [6, 1, 1, 3, 1, 1, 0, 0, 0, 0, 0, 6],
                    [6, 1, 1, 1, 1, 6, 6, 0, 0, 0, 0, 6],
                    [6, 1, 1, 1, 1, 6, 6, 0, 0, 0, 0, 6],
                    [6, 1, 1, 1, 1, 6, 6, 0, 0, 2, 4, 6],
                    [6, 1, 1, 1, 1, 6, 6, 0, 0, 0, 0, 6],
                    [6, 1, 5, 1, 1, 6, 6, 0, 0, 0, 0, 6],
                    [6, 1, 1, 1, 1, 6, 6, 0, 0, 0, 0, 6],
                    [6, 1, 1, 3, 1, 1, 0, 0, 0, 0, 0, 6],
                    [6, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 6],
                    [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                ]
            )

        self._field_map: Final[NDArray] = ctf_grid  # load_text_map(ctf_grid)
        height: int
        width: int
        height, width = self._field_map.shape

        self.obstacle: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["obstacle"]))
        )

        self.blue_flag: Final[Position] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_flag"]))
        )[0]

        self.red_flag: Final[Position] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_flag"]))
        )[0]

        self.blue_agent_pos: Final[list[Position]] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["blue_agent"]))
        )

        self.red_agent_pos: Final[Position] = list(
            zip(*np.where(self._field_map == self.world.OBJECT_TO_IDX["red_agent"]))
        )

        self.blue_territory: Final[list[Position]] = (
            list(
                zip(
                    *np.where(
                        self._field_map == self.world.OBJECT_TO_IDX["blue_territory"]
                    )
                )
            )
            + [self.blue_flag]
            + self.blue_agent_pos
        )

        self.red_territory: Final[list[Position]] = (
            list(
                zip(
                    *np.where(
                        self._field_map == self.world.OBJECT_TO_IDX["red_territory"]
                    )
                )
            )
            + [self.red_flag]
            + self.red_agent_pos
        )

        self.num_blue_agents = len(self.blue_agent_pos)
        self.num_red_agents = len(self.red_agent_pos)

        if self.num_blue_agents == 0:
            raise ValueError("***** Warning: No blue agents specified in the map *****")
        if self.num_red_agents == 0:
            print("***** Warning: No red agents specified in the map *****")

        blue_agents: list[AgentT] = [
            Agent(
                self.world,
                index=i,
                color="blue",
                bg_color="light_blue",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="blue_agent",
            )
            for i in range(self.num_blue_agents)
        ]

        # Check if there is only one policy for all enemy agents.
        if type(enemy_policies) is not list:
            enemy_policies = [enemy_policies for _ in range(self.num_red_agents)]
        else:
            # Check if the number of policies is equal to the number of enemy agents.
            assert len(enemy_policies) == self.num_red_agents

        if enemy_policy_kwargs is not list:
            enemy_policy_kwargs = [
                enemy_policy_kwargs for _ in range(self.num_red_agents)
            ]
        else:
            assert len(enemy_policy_kwargs) == self.num_red_agents

        # Initialize the enemy policies and set the random generator and field map.
        match enemy_policies[0]:
            case str():
                enemy_policies: list[CtfPolicyT] = [
                    HEURISTIC_POLICIES[policy](**kwargs)
                    for policy, kwargs in zip(enemy_policies, enemy_policy_kwargs)
                ]

            case _:
                enemy_policies: list[CtfPolicyT] = [
                    policy(**kwargs)
                    for policy, kwargs in zip(enemy_policies, enemy_policy_kwargs)
                ]

        for policy in enemy_policies:
            if hasattr(policy, "random_generator"):
                if policy.random_generator is None:
                    policy.random_generator = self.np_random
                else:
                    pass
            else:
                pass

            if hasattr(policy, "field_map"):
                if policy.field_map is None:
                    policy.field_map = self._field_map
                else:
                    pass
            else:
                pass

        red_agents: list[AgentT] = [
            PolicyAgent(
                enemy_policies[i],
                self.world,
                index=self.num_blue_agents + i,
                color="red",
                bg_color="light_red",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="red_agent",
            )
            for i in range(self.num_red_agents)
        ]

        agents: list[AgentT] = blue_agents + red_agents

        # Set the random generator and action set of the red agent from the env.
        for agent in agents:
            if type(agent) is PolicyAgent:
                agent.policy.random_generator = self.np_random
                agent.policy.action_set = self.actions_set
            else:
                pass

        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=agents,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            world=self.world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        self.action_space = (
            spaces.MultiDiscrete(
                [len(self.actions_set) for _ in range(self.num_blue_agents)]
            )
            if self.num_blue_agents > 1
            else spaces.Discrete(len(self.actions_set))
        )
        self.ac_dim = (
            self.action_space.shape if self.num_blue_agents > 1 else self.action_space.n
        )

        self.ep_game_stats: GameStats = {
            "defeated_blue_agents": 0,
            "defeated_red_agents": 0,
            "captured_blue_flags": 0,
            "captured_red_flags": 0,
            "blue_agent_defeated": [False for _ in range(self.num_blue_agents)],
            "red_agent_defeated": [False for _ in range(self.num_red_agents)],
            "blue_flag_captured": False,
            "red_flag_captured": False,
        }

    def _set_observation_space(self) -> spaces.Dict | spaces.Box:
        match self.observation_option:
            case "positional":
                observation_space = spaces.Dict(
                    {
                        "blue_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_blue_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_blue_agents)
                                ]
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "red_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_red_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_red_agents)
                                ]
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "blue_flag": spaces.Box(
                            low=np.array([0, 0]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "red_flag": spaces.Box(
                            low=np.array([0, 0]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "blue_territory": spaces.Box(
                            low=np.array(
                                [[0, 0] for _ in range(len(self.blue_territory))]
                            ).flatten(),
                            high=np.array(
                                np.array(
                                    [
                                        self._field_map.shape
                                        for _ in range(len(self.blue_territory))
                                    ]
                                )
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "red_territory": spaces.Box(
                            low=np.array(
                                [[0, 0] for _ in range(len(self.red_territory))]
                            ).flatten(),
                            high=np.array(
                                np.array(
                                    [
                                        self._field_map.shape
                                        for _ in range(len(self.red_territory))
                                    ]
                                )
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "obstacle": spaces.Box(
                            low=np.array(
                                [[0, 0] for _ in range(len(self.obstacle))]
                            ).flatten(),
                            high=np.array(
                                np.array(
                                    [
                                        self._field_map.shape
                                        for _ in range(len(self.obstacle))
                                    ]
                                )
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "terminated_agents": spaces.Box(
                            low=np.zeros(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                            high=np.ones(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                        ),
                    }
                )

            case "map":
                observation_space = spaces.Box(
                    low=0,
                    high=len(self.world.OBJECT_TO_IDX) - 1,
                    shape=self._field_map.shape,
                    dtype=np.int64,
                )

            case "flattened":
                obs_size: int = (
                    2 * (self.num_blue_agents + self.num_red_agents)
                    + 4  # 2 * (blue_flag + red_flag)
                    + 2 * len(self.obstacle)
                    + 2 * len(self.blue_territory)
                    + 2 * len(self.red_territory)
                    + self.num_blue_agents
                    + self.num_red_agents
                )
                obs_high = (
                    np.ones([obs_size])
                    * (np.max(self._field_map.shape) - 1)
                    / self.observation_scaling
                )
                obs_high[-(self.num_blue_agents + self.num_red_agents) :] = 1
                observation_space = spaces.Box(
                    low=np.zeros([obs_size]),
                    high=obs_high,
                    dtype=np.int64,
                )

            case "pos_map":
                observation_space = spaces.Dict(
                    {
                        "blue_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_blue_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_blue_agents)
                                ]
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "red_agent": spaces.Box(
                            low=np.array(
                                [[-1, -1] for _ in range(self.num_red_agents)]
                            ).flatten(),
                            high=np.array(
                                [
                                    self._field_map.shape
                                    for _ in range(self.num_red_agents)
                                ]
                            ).flatten()
                            - 1,
                            dtype=np.int64,
                        ),
                        "blue_flag": spaces.Box(
                            low=np.array([0, 0]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "red_flag": spaces.Box(
                            low=np.array([0, 0]),
                            high=np.array(self._field_map.shape) - 1,
                            dtype=np.int64,
                        ),
                        "static_map": spaces.Box(
                            low=0,
                            high=len(self.world.OBJECT_TO_IDX) - 1,
                            shape=self._field_map.shape,
                            dtype=np.int64,
                        ),
                        "terminated_agents": spaces.Box(
                            low=np.zeros(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                            high=np.ones(
                                [self.num_blue_agents + self.num_red_agents],
                                dtype=np.int64,
                            ),
                        ),
                    }
                )

            case "pos_map_flattened":
                obs_size: int = (
                    2 * (self.num_blue_agents + self.num_red_agents)
                    + 4  # 2 * (blue_flag + red_flag)
                    + np.prod(self._field_map.shape)
                    + self.num_blue_agents
                    + self.num_red_agents
                )
                obs_high = (
                    np.ones([obs_size])
                    * (np.max(self._field_map.shape) - 1)
                    / self.observation_scaling
                )
                obs_high[-(self.num_blue_agents + self.num_red_agents) :] = 1
                observation_space = spaces.Box(
                    low=np.zeros([obs_size]),
                    high=obs_high,
                    dtype=np.int64,
                )

            case "tensor":
                observation_space = spaces.Box(
                    low=0,
                    high=4,
                    shape=(self._field_map.shape[0], self._field_map.shape[1], 3),
                    dtype=np.int64,
                )

            case _:
                raise ValueError(
                    f"Invalid observation_option: {self.observation_option}"
                )

        return observation_space

    def _gen_grid(self, width, height, options):
        self.grid = Grid(width, height, self.world)

        for i, j in self.blue_territory:
            self.put_obj(
                Floor(self.world, color="light_blue", type="blue_territory"), i, j
            )

        for i, j in self.red_territory:
            self.put_obj(
                Floor(self.world, color="light_red", type="red_territory"), i, j
            )

        for i, j in self.obstacle:
            self.put_obj(Obstacle(self.world, penalty=self.obstacle_penalty), i, j)

        self.put_obj(
            Flag(
                self.world,
                index=0,
                color="blue",
                type="blue_flag",
                bg_color="light_blue",
            ),
            *self.blue_flag,
        )
        self.put_obj(
            Flag(
                self.world, index=1, color="red", type="red_flag", bg_color="light_red"
            ),
            *self.red_flag,
        )

        self.init_grid: Grid = self.grid.copy()

        # Choose non-overlapping indices for the blue agents and place them in the blue territory.
        if options["random_init_pos"]:
            combined_territory = self.blue_territory + self.red_territory
            indices = random.sample(
                range(len(combined_territory)),
                self.num_blue_agents + self.num_red_agents,
            )
            blue_agent_pos = [
                combined_territory[i] for i in indices[: self.num_blue_agents]
            ]
            red_agent_pos = [
                combined_territory[i] for i in indices[self.num_blue_agents :]
            ]
        else:
            blue_agent_pos = self.blue_agent_pos
            red_agent_pos = self.red_agent_pos

        for i in range(self.num_blue_agents):
            self.place_agent(
                self.agents[i],
                pos=blue_agent_pos[i],
                reset_agent_status=True,
            )

        # Choose non-overlapping indices for the red agents and place them in the red territory.
        if options["random_init_pos"]:
            indices = random.sample(range(len(self.red_territory)), self.num_red_agents)
            red_agent_pos = [self.red_territory[i] for i in indices]
        else:
            red_agent_pos = self.red_agent_pos

        for i in range(self.num_red_agents):
            self.place_agent(
                self.agents[self.num_blue_agents + i],
                pos=red_agent_pos[i],
                reset_agent_status=True,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict[str, float]]:
        self.game_stats: GameStats = {
            "defeated_blue_agents": 0,
            "defeated_red_agents": 0,
            "captured_blue_flags": 0,
            "captured_red_flags": 0,
            "blue_agent_defeated": [False for _ in range(self.num_blue_agents)],
            "red_agent_defeated": [False for _ in range(self.num_red_agents)],
            "blue_flag_captured": False,
            "red_flag_captured": False,
        }

        super().reset(seed=seed, options=options)

        self.blue_traj: list[list[Position]] = [
            [agent.pos] for agent in self.agents[0 : self.num_blue_agents]
        ]
        self.red_traj: list[list[Position]] = [
            [agent.pos] for agent in self.agents[self.num_blue_agents :]
        ]

        obs: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        return obs, info

    def _get_obs(self) -> Observation:
        observation: Observation

        match self.observation_option:
            case "positional":
                observation = self._get_dict_obs()
            case "map":
                observation = self._encode_map()
            case "flattened":
                observation = np.array(
                    [
                        *np.array(
                            [
                                agent.pos
                                for agent in self.agents[0 : self.num_blue_agents]
                            ]
                        ).flatten(),
                        *np.array(
                            [agent.pos for agent in self.agents[self.num_blue_agents :]]
                        ).flatten(),
                        *np.array(self.blue_flag),
                        *np.array(self.red_flag),
                        *np.array(self.blue_territory).flatten(),
                        *np.array(self.red_territory).flatten(),
                        *np.array(self.obstacle).flatten(),
                        *[int(agent.terminated) for agent in self.agents],
                    ]
                )
            case "pos_map":
                encoded_map: NDArray = np.zeros(self._field_map.shape, dtype=np.int64)

                for i, j in self.blue_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["blue_territory"]

                for i, j in self.red_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["red_territory"]

                for i, j in self.obstacle:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["obstacle"]

                observation: dict[str, NDArray | int] = {
                    "blue_agent": np.array(
                        [agent.pos for agent in self.agents[0 : self.num_blue_agents]]
                    ).flatten(),
                    "red_agent": np.array(
                        [agent.pos for agent in self.agents[self.num_blue_agents :]]
                    ).flatten(),
                    "blue_flag": np.array(self.blue_flag),
                    "red_flag": np.array(self.red_flag),
                    "static_map": encoded_map.T,
                    "terminated_agents": np.array(
                        [int(agent.terminated) for agent in self.agents]
                    ),
                }
            case "pos_map_flattened":
                encoded_map: NDArray = np.zeros(self._field_map.shape, dtype=np.int64)

                for i, j in self.blue_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["blue_territory"]

                for i, j in self.red_territory:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["red_territory"]

                for i, j in self.obstacle:
                    encoded_map[i, j] = self.world.OBJECT_TO_IDX["obstacle"]

                observation = np.array(
                    [
                        *np.array(
                            [
                                agent.pos
                                for agent in self.agents[0 : self.num_blue_agents]
                            ]
                        ).flatten(),
                        *np.array(
                            [agent.pos for agent in self.agents[self.num_blue_agents :]]
                        ).flatten(),
                        *np.array(self.blue_flag),
                        *np.array(self.red_flag),
                        *encoded_map.T.flatten(),
                        *[int(agent.terminated) for agent in self.agents],
                    ]
                )
            case "tensor":
                static_object_layer: NDArray[np.int_] = np.zeros(
                    self._field_map.shape, dtype=np.int_
                )
                agent_flag_layer: NDArray[np.int_] = np.zeros(
                    self._field_map.shape, dtype=np.int_
                )
                agent_status_layer: NDArray[np.int_] = np.zeros(
                    self._field_map.shape, dtype=np.int_
                )

                for i, j in self.blue_territory:
                    static_object_layer[i, j] = 1

                for i, j in self.red_territory:
                    static_object_layer[i, j] = 2

                for i, j in self.obstacle:
                    static_object_layer[i, j] = 0

                blue_agent_count: int = 0
                red_agent_count: int = 0
                for agent in self.agents[0 : self.num_blue_agents]:
                    assert agent.pos is not None
                    agent_flag_layer[agent.pos[0], agent.pos[1]] = 1
                    agent_status_layer[agent.pos[0], agent.pos[1]] = (
                        1 if agent.terminated else 2
                    )

                    blue_agent_count += 1

                for agent in self.agents[self.num_blue_agents :]:
                    assert agent.pos is not None
                    agent_flag_layer[agent.pos[0], agent.pos[1]] = 2
                    agent_status_layer[agent.pos[0], agent.pos[1]] = (
                        1 if agent.terminated else 2
                    )

                    red_agent_count += 1

                if blue_agent_count != self.num_blue_agents:
                    raise ValueError(
                        f"Number of blue agents in the grid ({blue_agent_count}) is not equal to the number of blue agents ({self.num_blue_agents}) at step {self.step_count}."
                    )
                elif red_agent_count != self.num_red_agents:
                    raise ValueError(
                        f"Number of red agents in the grid ({red_agent_count}) is not equal to the number of red agents ({self.num_red_agents}) at step {self.step_count}."
                    )
                else:
                    pass

                # Flags
                agent_flag_layer[self.blue_flag[0], self.blue_flag[1]] = 3
                agent_flag_layer[self.red_flag[0], self.red_flag[1]] = 4

                observation = np.stack(
                    [static_object_layer.T, agent_flag_layer.T, agent_status_layer.T],
                    axis=2,
                )

            case _:
                raise ValueError(
                    f"Invalid observation_option: {self.observation_option}"
                )

        return observation

    def _get_dict_obs(self) -> ObservationDict:
        for a in self.agents:
            assert a.pos is not None

        observation: ObservationDict

        observation = {
            "blue_agent": np.array(
                [agent.pos for agent in self.agents[0 : self.num_blue_agents]]
            ).flatten(),
            "red_agent": np.array(
                [agent.pos for agent in self.agents[self.num_blue_agents :]]
            ).flatten(),
            "blue_flag": np.array(self.blue_flag),
            "red_flag": np.array(self.red_flag),
            "blue_territory": np.array(self.blue_territory).flatten(),
            "red_territory": np.array(self.red_territory).flatten(),
            "obstacle": np.array(self.obstacle).flatten(),
            "terminated_agents": np.array(
                [int(agent.terminated) for agent in self.agents]
            ),
        }

        return observation

    def _encode_map(self) -> NDArray:
        encoded_map: NDArray = np.zeros(self._field_map.shape, dtype=np.int64)

        for i, j in self.blue_territory:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["blue_territory"]

        for i, j in self.red_territory:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["red_territory"]

        for i, j in self.obstacle:
            encoded_map[i, j] = self.world.OBJECT_TO_IDX["obstacle"]

        encoded_map[self.blue_flag[0], self.blue_flag[1]] = self.world.OBJECT_TO_IDX[
            "blue_flag"
        ]

        encoded_map[self.red_flag[0], self.red_flag[1]] = self.world.OBJECT_TO_IDX[
            "red_flag"
        ]

        for agent in self.agents:
            assert agent.pos is not None
            encoded_map[agent.pos[0], agent.pos[1]] = self.world.OBJECT_TO_IDX[
                agent.type if not agent.terminated else "obstacle"
            ]

        return encoded_map.T

    def _get_info(self) -> dict[str, float]:
        assert self.agents[0].pos is not None
        assert self.agents[1].pos is not None

        # self.game_stats["red_flag_captured"] = True
        # self.game_stats["captured_red_flags"] += 1

        info = {
            "success": self.game_stats["red_flag_captured"],
            "failure": self.game_stats["blue_agent_defeated"][0],
            "d_ba_ra": distance_points(self.agents[0].pos, self.agents[1].pos),
            "d_ba_bf": distance_points(self.agents[0].pos, self.blue_flag),
            "d_ba_rf": distance_points(self.agents[0].pos, self.red_flag),
            "d_ra_bf": distance_points(self.agents[1].pos, self.blue_flag),
            "d_ra_rf": distance_points(self.agents[1].pos, self.red_flag),
            "d_bf_rf": distance_points(self.blue_flag, self.red_flag),
            "d_ba_bt": distance_area_point(self.agents[0].pos, self.blue_territory),
            "d_ba_rt": distance_area_point(self.agents[0].pos, self.red_territory),
            "d_ra_bt": distance_area_point(self.agents[1].pos, self.blue_territory),
            "d_ra_rt": distance_area_point(self.agents[1].pos, self.red_territory),
            "d_ba_ob": distance_area_point(self.agents[0].pos, self.obstacle),
            "d_ra_ob": distance_area_point(self.agents[1].pos, self.obstacle),
            "d_ra_df": (
                1
                if self.num_red_agents == self.game_stats["defeated_red_agents"]
                else -1
            ),
        } | self.ep_game_stats
        return info

    def _move_agent(self, action: int, agent: AgentT) -> None:
        next_pos: Position

        assert agent.pos is not None

        match action:
            case self.actions_set.stay:
                next_pos = agent.pos
            case self.actions_set.left:
                next_pos = agent.pos + np.array([0, -1])
            case self.actions_set.down:
                next_pos = agent.pos + np.array([-1, 0])
            case self.actions_set.right:
                next_pos = agent.pos + np.array([0, 1])
            case self.actions_set.up:
                next_pos = agent.pos + np.array([1, 0])
            case _:
                raise ValueError(f"Invalid action: {action}")

        if (
            next_pos[0] < 0
            or next_pos[1] < 0
            or next_pos[0] >= self.width
            or next_pos[1] >= self.height
        ):
            pass
        else:
            next_cell: WorldObjT | None = self.grid.get(*next_pos)

            is_agent_in_blue_territory: bool = self._is_agent_in_territory(
                agent.type, "blue", next_pos
            )
            is_agent_in_red_territory: bool = self._is_agent_in_territory(
                agent.type, "red", next_pos
            )

            if is_agent_in_blue_territory:
                bg_color = "light_blue"
            elif is_agent_in_red_territory:
                bg_color = "light_red"
            else:
                bg_color = None

            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            elif next_cell.can_overlap():
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            elif self.obstacle_penalty != 0 and (
                next_cell.type == "obstacle"
                or next_cell.type == "red_agent"
                or next_cell.type == "blue_agent"
            ):
                agent.collided = True
            else:
                pass

    def _move_agents(self, actions: list[int]) -> None:
        # Randomly generate the order of the agents by indices using self.np_random.
        agent_indices: list[int] = list(
            range(self.num_blue_agents + self.num_red_agents)
        )
        self.np_random.shuffle(agent_indices)
        for i in agent_indices:
            if self.agents[i].terminated:
                # Defeated agent doesn't move, sadly.
                pass
            else:
                self._move_agent(actions[i], self.agents[i])

    def _is_agent_in_territory(
        self,
        agent_type: Literal["blue_agent", "red_agent"],
        territory_name: Literal["blue", "red"],
        agent_loc: Position | None = None,
    ) -> bool:
        in_territory: bool = False

        territory: list[Position]
        if agent_loc is None:
            match agent_type:
                case "blue_agent":
                    assert self.agents[0].pos is not None
                    agent_loc = self.agents[0].pos
                case "red_agent":
                    assert self.agents[1].pos is not None
                    agent_loc = self.agents[1].pos
                case _:
                    raise ValueError(f"Invalid agent_name: {agent_type}")
        else:
            pass

        match territory_name:
            case "blue":
                territory = self.blue_territory
            case "red":
                territory = self.red_territory
            case _:
                raise ValueError(f"Invalid territory_name: {territory_name}")

        for i, j in territory:
            if agent_loc[0] == i and agent_loc[1] == j:
                in_territory = True
                break
            else:
                pass

        return in_territory

    def step(
        self, blue_actions: list[int] | int
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        self.step_count += 1

        blue_actions: NDArray[np.int_] = np.array(blue_actions).flatten()

        red_actions: list[int] = []
        for red_agent in self.agents[self.num_blue_agents :]:
            assert type(red_agent) is PolicyAgent
            red_action: int = red_agent.policy.act(self._get_dict_obs(), red_agent.pos)
            red_actions.append(red_action)

        # Just in case NN outputs are, for some reason, not discrete.
        rounded_blue_actions: NDArray[np.int_] = np.round(blue_actions).astype(np.int_)
        # Concatenate the blue and red actions as 1D array
        actions: Final[list[int]] = rounded_blue_actions.tolist() + red_actions

        self._move_agents(actions)

        terminated: bool = False
        truncated: Final[bool] = self.step_count >= self.max_steps

        reward: float = 0.0

        # Calculate the collision penalty for the blue agents.
        if self.obstacle_penalty != 0:
            for blue_agent in self.agents[0 : self.num_blue_agents]:
                if blue_agent.collided:
                    reward -= self.obstacle_penalty
                    blue_agent.terminated = True
                    blue_agent.color = "blue_grey"
                else:
                    pass

            for red_agent in self.agents[self.num_blue_agents :]:
                if red_agent.collided:
                    red_agent.terminated = True
                    red_agent.color = "red_grey"
                else:
                    pass
        else:
            pass

        # Calculate the flag rewards of the blue agents
        for blue_agent in self.agents[0 : self.num_blue_agents]:
            if (
                blue_agent.pos[0] == self.red_flag[0]
                and blue_agent.pos[1] == self.red_flag[1]
            ):
                reward += self.flag_reward
                terminated = True
                self.game_stats["red_flag_captured"] = True
                self.game_stats["captured_red_flags"] += 1
            else:
                pass

        # Calculate the flag rewards (penalties) of the red agents
        for red_agent in self.agents[self.num_blue_agents :]:
            if (
                red_agent.pos[0] == self.blue_flag[0]
                and red_agent.pos[1] == self.blue_flag[1]
            ):
                reward -= self.flag_reward
                terminated = True
                self.game_stats["blue_flag_captured"] = True
                self.game_stats["captured_blue_flags"] += 1
            else:
                pass

        # Calculate the distances between the blue and red agents and the battle outcomes if they are within the battle range.
        blue_agent_locs: list[Position] = [
            agent.pos for agent in self.agents[0 : self.num_blue_agents]
        ]
        red_agent_locs: list[Position] = [
            agent.pos for agent in self.agents[self.num_blue_agents :]
        ]
        blue_agent_locs_np: NDArray[np.float64] = np.array(blue_agent_locs)
        red_agent_locs_np: NDArray[np.float64] = np.array(red_agent_locs)

        distances: NDArray[np.float64] = np.linalg.norm(
            blue_agent_locs_np[:, np.newaxis] - red_agent_locs_np, axis=2
        )

        # Get the indices of distances that are less than the battle range.
        battle_indices: tuple[NDArray[np.int_], NDArray[np.int_]] = np.where(
            distances <= self.battle_range
        )
        # Iterate over the indices and perform the battles.
        for blue_agent_idx, red_agent_idx in zip(*battle_indices):
            # Battle only takes place if both agents are not defeated (terminated).
            if (
                not self.agents[blue_agent_idx].terminated
                and not self.agents[self.num_blue_agents + red_agent_idx].terminated
            ):
                blue_agent_in_blue_territory: bool = self._is_agent_in_territory(
                    "blue_agent", "blue", blue_agent_locs[blue_agent_idx]
                )
                red_agent_in_red_territory: bool = self._is_agent_in_territory(
                    "red_agent", "red", red_agent_locs[red_agent_idx]
                )

                blue_win: bool
                match (blue_agent_in_blue_territory, red_agent_in_red_territory):
                    case (True, True):
                        blue_win = self.np_random.choice([True, False])
                    case (True, False):
                        blue_win = self.np_random.choice([True, True])
                    case (False, True):
                        blue_win = self.np_random.choice([False, False])
                    case (False, False):
                        blue_win = self.np_random.choice([True, False])
                    case (_, _):
                        raise ValueError(
                            f"Invalid combination of blue_agent_in_blue_territory: {blue_agent_in_blue_territory} and red_agent_in_red_territory: {red_agent_in_red_territory}"
                        )

                if blue_win:
                    reward += self.battle_reward
                    self.agents[self.num_blue_agents + red_agent_idx].terminated = True
                    self.agents[self.num_blue_agents + red_agent_idx].color = "red_grey"
                    self.game_stats["red_agent_defeated"][red_agent_idx] = True
                    self.game_stats["defeated_red_agents"] += 1
                else:
                    reward -= self.battle_reward
                    self.agents[blue_agent_idx].terminated = True
                    self.agents[blue_agent_idx].color = "blue_grey"
                    self.game_stats["blue_agent_defeated"][blue_agent_idx] = True
                    self.game_stats["defeated_blue_agents"] += 1
            else:
                pass

        # Check if all the blue agents are defeated.
        if all(agent.terminated for agent in self.agents[0 : self.num_blue_agents]):
            terminated = True
        else:
            pass

        reward -= self.step_penalty

        observation: Observation = self._get_obs()
        info: dict[str, float] = self._get_info()

        if terminated or truncated:
            self.ep_game_stats = self.game_stats
        else:
            pass

        return observation, reward, terminated, truncated, info


class CtF(CtfMvNEnv):
    """
    Environment for capture the flag game with one ego (blue) agent and one enemy (red) agent.
    """

    def __init__(
        self,
        grid_type: int,
        max_steps: int,
        enemy_policy: Type[CtfPolicyT] | str = RoombaPolicy,
        enemy_policy_kwarg: dict[str, Any] = {},
        battle_range: float = 1,
        territory_adv_rate: float = 1.0,
        flag_reward: float = 1,
        battle_reward_ratio: float = 0.5,
        obstacle_penalty_ratio: float = 0,
        step_penalty_ratio: float = 0.01,
        observation_option: ObservationOption = "positional",
        observation_scaling: float = 1,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
        uncached_object_types: list[str] = ["red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the environment.

        Parameters
        ----------
        map_path : str
            Path to the map file.
        enemy_policies : CtfPolicyT = RwPolicy()
            Policy of the enemy agent.
        enemy_policy_avoided_objects : list[str] = ["obstacle", "blue_agent", "red_agent"]
            Types of objects that the enemy agent should avoid in the path.
            The object names should match with those in the environment's world object.
        battle_range : float = 1
            Range within which battles can occur.
        territory_adv_rate : float = 0.75
            Probability of the enemy agent winning a battle within its territory.
        flag_reward : float = 1
            Reward for capturing the enemy flag.
        battle_reward_ratio : float = 0.25
            Ratio of the flag reward for winning a battle.
        obstacle_penalty_ratio : float = 0
            Ratio of the flag reward for colliding with an obstacle.
        step_penalty_ratio : float = 0.01
            Ratio of the flag reward for taking a step.
        max_steps : int = 100
            Maximum number of steps per episode.
        observation_option : ObservationOption = "positional"
            Observation option.
        observation_scaling : float = 1
            Scaling factor for the observation.
        render_mode : Literal["human", "rgb_array"] = "rgb_array"
            Rendering mode.
        uncached_object_types : list[str] = ["red_agent", "blue_agent"]
            Types of objects that should not be cached.
        """
        super().__init__(
            grid_type=grid_type,
            enemy_policies=enemy_policy,
            enemy_policy_kwargs=enemy_policy_kwarg,
            battle_range=battle_range,
            territory_adv_rate=territory_adv_rate,
            flag_reward=flag_reward,
            battle_reward_ratio=battle_reward_ratio,
            obstacle_penalty_ratio=obstacle_penalty_ratio,
            step_penalty_ratio=step_penalty_ratio,
            max_steps=max_steps,
            observation_option=observation_option,
            observation_scaling=observation_scaling,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )
