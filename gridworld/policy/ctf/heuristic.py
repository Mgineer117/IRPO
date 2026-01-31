from enum import IntEnum
from typing import Literal, Type, TypeVar

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from gridworld.core.agent import ActionsT, GridActions
from gridworld.core.world import CtfWorld, WorldT
from gridworld.policy.base import BaseAgentPolicy
from gridworld.policy.ctf.typing import ObservationDict
from gridworld.policy.ctf.utils import a_star, get_unterminated_opponent_pos
from gridworld.typing import Position
from gridworld.utils.map import (
    closest_area_pos,
    distance_area_point,
    position_in_positions,
)

CtfPolicyT = TypeVar("CtfPolicyT", bound="CtfPolicy")


class CtfPolicy(BaseAgentPolicy):
    """
    Abstract class for Capture the Flag agent policy
    """

    def __init__(
        self,
        action_set: type[IntEnum] | None = None,
        random_generator: Generator | None = None,
    ) -> None:
        super().__init__(action_set, random_generator)
        self.name: str = "ctf"

    def act(self, observation: ObservationDict, curr_pos: Position) -> int:
        """
        Determine the action to take.

        Parameters
        ----------
        observation : ObservationDict
            Observation dictionary (dict from the env).
        curr_pos : Position
            Current position of the agent.

        Returns
        -------
        int
            Action to take.
        """
        raise NotImplementedError

    def act_randomly(self) -> int:
        """
        Determine the action to take randomly.

        Returns
        -------
        int
            Action to take.
        """
        return self.random_generator.integers(0, len(self.action_set))

    def dir_to_action(self, action_dir: NDArray[np.int_]) -> int:
        """
        Convert the direction to an action.

        Parameters
        ----------
        action_dir : NDArray[np.int_]
            Direction to convert to an action.

        Returns
        -------
        int
            Action to take.
        """

        action: int

        if np.array_equal(action_dir, np.array([0, 0])):
            action = self.action_set.stay
        elif np.array_equal(action_dir, np.array([0, -1])):
            action = self.action_set.left
        elif np.array_equal(action_dir, np.array([-1, 0])):
            action = self.action_set.down
        elif np.array_equal(action_dir, np.array([0, 1])):
            action = self.action_set.right
        elif np.array_equal(action_dir, np.array([1, 0])):
            action = self.action_set.up
        else:
            raise ValueError(f"Invalid direction {action_dir}")

        return action

    def action_to_dir(self, action: int) -> NDArray[np.int_]:
        """
        Convert the action to a direction.

        Parameters
        ----------
        action : int
            Action to convert to a direction.

        Returns
        -------
        NDArray[np.int_]
            Direction.
        """

        action_dir: NDArray[np.int_]

        if action == self.action_set.stay:
            action_dir = np.array([0, 0])
        elif action == self.action_set.left:
            action_dir = np.array([0, -1])
        elif action == self.action_set.down:
            action_dir = np.array([-1, 0])
        elif action == self.action_set.right:
            action_dir = np.array([0, 1])
        elif action == self.action_set.up:
            action_dir = np.array([1, 0])
        else:
            raise ValueError(f"Invalid action {action}")

        return action_dir

    def reset(self) -> None:
        pass


class RwPolicy(CtfPolicy):
    """
    Random walk policy

    Attributes:
        name: str
            Policy name
        random_generator: numpy.random.Generator
            Random number generator.
            Replace it with the environment's random number generator if needed.
    """

    def __init__(
        self,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
    ) -> None:
        """
        Initialize the RW policy.

        Parameters
        ----------
        action_set : gym_multigrid.core.agent.ActionsT | None
            Actions available to the agent.
        random_generator : numpy.random.Generator
            Random number generator.
            Replace it with the environment's random number generator if needed.
        """
        super().__init__(action_set, random_generator)
        self.name = "rw"

    def act(
        self,
        observation: ObservationDict | None = None,
        curr_pos: Position | None = None,
    ) -> int:
        return self.act_randomly()


class DestinationPolicy(CtfPolicy):
    """
    Policy that always tries to reach a destination with possible randomness in action selection.

    Attributes:
        name: str
            Policy name
    """

    def __init__(
        self,
        field_map: NDArray | None = None,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
        randomness: float = 0.25,
        world: WorldT = CtfWorld,
        avoided_objects: list[str] = ["obstacle", "red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
            Make sure to set it to the field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = GridActions
            Actions available to the agent.
        random_generator : numpy.random.Generator | None = None
            Random number generator. Replace it with the environment's random number generator if needed.
        randomness : float
            Probability of taking an random action instead of an optimal action.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied, and it should be set to the environment's world object.
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world object.
        """
        super().__init__(action_set, random_generator)
        self.name = "destination"
        self.field_map: NDArray | None = field_map
        self.randomness: float = randomness
        self.world: WorldT = world
        self.avoided_objects: list[str] = avoided_objects

    def get_target(self, observation: ObservationDict, curr_pos: Position) -> Position:
        """
        Get the target position of the agent.

        Parameters
        ----------
        observation : ObservationDict
            Observation dictionary (dict from the env).

        Returns
        -------
        Position
            Target position of the agent.
        """
        # Implement this method
        raise NotImplementedError("Implement the get_target method")

    def act(self, observation: ObservationDict, curr_pos: Position) -> int:
        """
        Determine the action to take.

        Parameters
        ----------
        observation : ObservationDict
            Observation dictionary (dict from the env).
        curr_pos : Position
            Current position of the agent.

        Returns
        -------
        int
            Action to take.
        """

        start_np: NDArray = np.array(curr_pos)
        target_np: NDArray = np.array(self.get_target(observation, curr_pos))
        # Convert start and target to tuple from NDArray
        start: Position = tuple(start_np)
        target: Position = tuple(target_np)
        shortest_path = a_star(
            start, target, self.field_map, self.world, self.avoided_objects
        )
        optimal_loc: Position = np.array(
            shortest_path[1] if len(shortest_path) > 1 else target
        )

        # Determine if the agent should take the optimal action
        is_action_optimal: bool = self.random_generator.choice(
            [True, False], p=[1 - self.randomness, self.randomness]
        )

        # If the optimal action is not taken, return a random action
        action: int
        if is_action_optimal:
            action_dir: NDArray = np.array(optimal_loc) - start_np
            action = self.dir_to_action(action_dir)
        else:
            action = self.act_randomly()

        return action


class FightPolicy(DestinationPolicy):
    """
    Policy that always tries to fight by taking the shortest path to the opponent agent with some stochasticity.

    Attributes:
        name: str
            Policy name
    """

    def __init__(
        self,
        field_map: NDArray | None = None,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
        randomness: float = 0.25,
        ego_agent: Literal["red", "blue"] = "red",
        world: WorldT = CtfWorld,
        avoided_objects: list[str] = ["obstacle", "red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = GridActions
            Actions available to the agent.
        randomness : float = 0.25
            Probability of taking an random action instead of an optimal action.
        ego_agent : Literal["red", "blue"] = "red"
            Controlled agent.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied.
            It should be set to the environment's world object.
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world object.
        """

        super().__init__(
            field_map, action_set, random_generator, randomness, world, avoided_objects
        )
        self.name = "fight"
        self.ego_agent: Literal["red", "blue"] = ego_agent

    def get_target(self, observation: ObservationDict, curr_pos: Position) -> Position:
        opponent_agent: Literal["red_agent", "blue_agent"] = (
            "blue_agent" if self.ego_agent == "red" else "red_agent"
        )
        opponent_pos: list[Position] = get_unterminated_opponent_pos(
            observation, opponent_agent
        )

        target: Position = closest_area_pos(curr_pos, opponent_pos)

        return target


class CapturePolicy(DestinationPolicy):
    """
    Policy that always tries to capture the flag by taking the shortest path to the opponent flag with some stochasticity.

    Attributes:
        name: str
            Policy name
    """

    def __init__(
        self,
        field_map: NDArray | None = None,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
        randomness: float = 0.25,
        ego_agent: Literal["red", "blue"] = "red",
        world: WorldT = CtfWorld,
        avoided_objects: list[str] = ["obstacle", "red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = GridActions
            Actions available to the agent.
        randomness : float = 0.25
            Probability of taking an random action instead of an optimal action.
        ego_agent : Literal["red", "blue"] = "red"
            Controlled agent.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied.
            It should be set to the environment's world object.
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world
        """

        super().__init__(
            field_map, action_set, random_generator, randomness, world, avoided_objects
        )
        self.name = "capture"
        self.ego_agent: Literal["red", "blue"] = ego_agent

    def get_target(self, observation: ObservationDict, curr_pos: Position) -> Position:
        match self.ego_agent:
            case "red":
                assert "blue_flag" in observation
                return observation["blue_flag"]
            case "blue":
                assert "red_flag" in observation
                return observation["red_flag"]


class PatrolPolicy(DestinationPolicy):
    """
    Policy that always tries to patrol around the border between blue and red territories with some stochasticity.

    Attributes:
        name: str
            Policy name
    """

    def __init__(
        self,
        field_map: NDArray | None = None,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
        randomness: float = 0.25,
        ego_agent: Literal["red", "blue"] = "red",
        world: WorldT = CtfWorld,
        avoided_objects: list[str] = ["obstacle", "red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = GridActions
            Actions available to the agent.
        randomness : float = 0.25
            Probability of taking an random action instead of an optimal action.
        ego_agent : Literal["red", "blue"] = "red"
            Controlled agent.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied.
            It should be set to the environment's world object.
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world object.
        """

        super().__init__(
            field_map, action_set, random_generator, randomness, avoided_objects
        )
        self.name = "patrol"
        self.ego_agent: Literal["red", "blue"] = ego_agent
        self.world: WorldT = world

        self.directions: list[Position] = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        self.border: list[Position]
        self.obstacle: list[Position]
        self.border, self.obstacle = self.locate_border(world, self.directions)

    def get_target(self, observation: ObservationDict, curr_pos: Position) -> Position:

        if position_in_positions(curr_pos, self.border):
            possible_next_pos: list[Position] = [
                (pos[0] + dir[0], pos[1] + dir[1])
                for pos in self.border
                for dir in self.directions
            ]
            optimal_locs: list[Position] = [
                pos
                for pos in possible_next_pos
                if position_in_positions(pos, self.border)
            ]
            target: Position = self.random_generator.choice(optimal_locs)
        else:
            target: Position = closest_area_pos(curr_pos, self.border)

        return target

    def locate_border(
        self, world: WorldT, directions: list[Position]
    ) -> tuple[list[Position], list[Position]]:
        """
        Locate the border between red and blue territories.

        Parameters
        ----------
        ego_agent : Literal["red", "blue"]
            Controlled agent.

        Returns
        -------
        border: list[Position]
            List of positions on the border.
        """

        assert self.world is not None

        own_territory_type: str = (
            "red_territory" if self.ego_agent == "red" else "blue_territory"
        )
        opponent_territory_type: str = (
            "red_territory" if self.ego_agent == "blue" else "blue_territory"
        )

        own_territory: list[Position] = list(
            zip(*np.where(self.field_map == world.OBJECT_TO_IDX[own_territory_type]))
        )
        opponent_territory: list[Position] = list(
            zip(
                *np.where(
                    self.field_map == world.OBJECT_TO_IDX[opponent_territory_type]
                )
            )
        )
        obstacle: list[Position] = list(
            zip(*np.where(self.field_map == world.OBJECT_TO_IDX["obstacle"]))
        )

        border: list[Position] = []

        for loc in own_territory:
            for dir in directions:
                new_loc: Position = (loc[0] + dir[0], loc[1] + dir[1])
                if position_in_positions(new_loc, opponent_territory + obstacle):
                    border.append(new_loc)
                    break
                else:
                    pass

        return border, obstacle


class PatrolFightPolicy(PatrolPolicy):
    """
    Policy that always tries to patrol around the border between blue and red territories and, once the opponent agent enters the ego territory, it tries to fight by taking the shortest path the opponent.

    Attributes:
        name: str
            Policy name
    """

    def __init__(
        self,
        field_map: NDArray | None = None,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
        randomness: float = 0.25,
        ego_agent: Literal["red", "blue"] = "red",
        world: WorldT = CtfWorld,
        avoided_objects: list[str] = ["obstacle", "red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = GridActions
            Actions available to the agent.
        randomness : float = 0.25
            Probability of taking an random action instead of an optimal action.
        ego_agent : Literal["red", "blue"] = "red"
            Controlled agent.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied.
            It should be set to the environment's world
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world object.
        """

        super().__init__(
            field_map,
            action_set,
            random_generator,
            randomness,
            ego_agent,
            world,
            avoided_objects,
        )
        self.name = "patrol_fight"

    def get_target(self, observation: ObservationDict, curr_pos: Position) -> Position:
        opponent_agent: Literal["red_agent", "blue_agent"] = (
            "blue_agent" if self.ego_agent == "red" else "red_agent"
        )
        ego_territory: Literal["red_territory", "blue_territory"] = (
            "red_territory" if self.ego_agent == "red" else "blue_territory"
        )

        opponent_pos: list[Position] = get_unterminated_opponent_pos(
            observation, opponent_agent
        )

        ego_territory_pos_np: NDArray = observation[ego_territory].reshape(-1, 2)
        ego_territory_pos: list[Position] = [tuple(pos) for pos in ego_territory_pos_np]

        # Check if the opponent agent is in the ego territory
        is_opponent_in_ego_territory: bool = False
        for pos in opponent_pos:
            if position_in_positions(pos, ego_territory_pos):
                is_opponent_in_ego_territory = True
                break
            else:
                pass

        target: Position = (
            closest_area_pos(curr_pos, opponent_pos)
            if is_opponent_in_ego_territory
            else super().get_target(observation, curr_pos)
        )

        return target


class RoombaPolicy(CtfPolicy):
    """
    Policy that always tries to roam around the environment with some stochasticity.
    This method generate an action for given agent.
    Agent is given with limited vision of a field.
    This method provides simple protocol of movement based on the agent's location and it's vision.

    Protocol
    --------
    1. Scan the area with flag_range:
        - Flag within radius : Set the movement towards flag
        - No Flag : random movement
    2. Scan the area with enemy_range:
        - Enemy in the direction of movement
            - If I'm in enemy's territory: reverse direction
            - Else chase the enemy.
        - Else: continue moving in the direction
    3. Random exploration
        - 0.1 chance switching direction of movement
        - Follow same direction
        - Change direction if it heats the wall

    Attributes
    ----------
        name: str
            Policy name
    """

    def __init__(
        self,
        enemy_range: int = 2,
        flag_range: int = 3,
        field_map: NDArray | None = None,
        action_set: ActionsT = GridActions,
        random_generator: Generator | None = None,
        randomness: float = 0.15,
        ego_agent: Literal["red", "blue"] = "red",
        world: WorldT = CtfWorld,
        avoided_objects: list[str] = ["obstacle", "red_agent", "blue_agent"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        enemy_range : int = 2
            Range to detect the enemy.
        flag_range : int = 3
            Range to detect the flag.
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
            Make sure to set it to the field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = GridActions
            Actions available to the agent.
        random_generator : numpy.random.Generator | None = None
            Random number generator.
            Replace it with the environment's random number generator if needed.
        randomness : float = 0.25
            Probability of taking a random action instead of an optimal action.
        ego_agent : Literal["red", "blue"] = "red"
            Controlled agent.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied.
            It should be set to the environment's world object.
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world object.
        """

        super().__init__(action_set, random_generator)
        self.field_map: NDArray | None = field_map
        self.randomness: float = randomness
        self.world: WorldT = world
        self.avoided_objects: list[str] = avoided_objects
        self.name = "roomba"
        self.enemy_range: int = enemy_range
        self.flag_range: int = flag_range
        self.ego_agent: Literal["red_agent", "blue_agent"] = ego_agent + "_agent"
        self.previous_action: int = self.act_randomly()

    def reset(self) -> None:
        self.previous_action = self.act_randomly()

    def act(self, observation: ObservationDict, curr_pos: Position) -> int:
        """
        Determine the action to take.

        Parameters
        ----------
        observation : ObservationDict
            Observation dictionary (dict from the env).
        curr_pos : Position
            Current position of the agent.

        Returns
        -------
        int
            Action to take.
        """

        # Determine if the agent should take the optimal action or a random action
        is_action_optimal: bool = self.random_generator.choice(
            [True, False], p=[1 - self.randomness, self.randomness]
        )

        if is_action_optimal:
            # 1. Scan the area with flag_range
            flag_pos: NDArray[np.int_] = (
                observation["blue_flag"]
                if self.ego_agent == "red_agent"
                else observation["red_flag"]
            )
            flag_dist: int = np.linalg.norm(np.array(flag_pos) - np.array(curr_pos))

            if flag_dist <= self.flag_range:
                start_np: NDArray = np.array(curr_pos)
                target_np: NDArray = flag_pos
                # Convert start and target to tuple from NDArray
                start: Position = tuple(start_np)
                target: Position = tuple(target_np)
                shortest_path = a_star(
                    start, target, self.field_map, self.world, self.avoided_objects
                )
                optimal_loc: Position = np.array(
                    shortest_path[1] if len(shortest_path) > 1 else target
                )

                # Determine if the agent should take the optimal action
                is_action_optimal: bool = self.random_generator.choice(
                    [True, False], p=[1 - self.randomness, self.randomness]
                )

                # If the optimal action is not taken, return a random action
                action_dir: NDArray = np.array(optimal_loc) - start_np
                action = self.dir_to_action(action_dir)

            else:
                action = self.act_randomly()

            # 2. Scan the area with enemy_range
            opponent_agent: Literal["red_agent", "blue_agent"] = (
                "blue_agent" if self.ego_agent == "red_agent" else "red_agent"
            )
            opponent_pos: list[Position] = []

            for op_pos in get_unterminated_opponent_pos(observation, opponent_agent):
                opp_dist: int = np.linalg.norm(np.array(op_pos) - np.array(curr_pos))
                if opp_dist <= self.enemy_range:
                    opponent_pos.append(op_pos)
                else:
                    pass

            new_pos: NDArray[np.int_] = np.array(curr_pos) + self.action_to_dir(action)
            if len(opponent_pos) > 0:
                opponent_territory: NDArray[np.int_] = (
                    observation["red_territory"]
                    if opponent_agent == "red_agent"
                    else observation["blue_territory"]
                ).reshape(-1, 2)
                if position_in_positions(curr_pos, opponent_territory):
                    if distance_area_point(new_pos, opponent_pos) < distance_area_point(
                        curr_pos, opponent_pos
                    ):
                        action = self.dir_to_action(
                            np.array(curr_pos) - np.array(new_pos)
                        )
                    else:
                        pass
                else:
                    # If the agent is its own territory, chase the opponent
                    target: Position = closest_area_pos(curr_pos, opponent_pos)
                    # Convert start and target to tuple from NDArray
                    start: Position = curr_pos
                    shortest_path = a_star(
                        tuple(start),
                        tuple(target),
                        self.field_map,
                        self.world,
                        self.avoided_objects,
                    )
                    optimal_loc: Position = (
                        shortest_path[1] if len(shortest_path) > 1 else target
                    )
                    action_dir: NDArray = np.array(optimal_loc) - np.array(curr_pos)
                    action = self.dir_to_action(action_dir)

            else:
                pass

            # Check if there is an obstacle in the direction of movement
            if (
                self.world.IDX_TO_OBJECT[self.field_map[new_pos[0], new_pos[1]]]
                == "obstacle"
            ):
                # Select an action that is not in the direction of the obstacle
                available_actions: list[int] = []
                for act in range(len(self.action_set)):
                    dir_tmp: NDArray = self.action_to_dir(act)
                    new_pos_tmp: NDArray = np.array(curr_pos) + dir_tmp
                    if (
                        self.world.IDX_TO_OBJECT[
                            self.field_map[new_pos_tmp[0], new_pos_tmp[1]]
                        ]
                        not in self.avoided_objects
                    ):
                        available_actions.append(act)
                    else:
                        pass
                action = self.random_generator.choice(available_actions)
            else:
                pass

        else:
            action = self.act_randomly()  # 3. Random exploration

        self.previous_action = action

        return action


HEURISTIC_POLICIES: dict[str, Type[CtfPolicyT]] = {
    "rw": RwPolicy,
    "fight": FightPolicy,
    "capture": CapturePolicy,
    "patrol": PatrolPolicy,
    "patrol_fight": PatrolFightPolicy,
    "roomba": RoombaPolicy,
}
