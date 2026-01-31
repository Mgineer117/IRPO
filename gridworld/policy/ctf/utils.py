from heapq import heapify, heappop, heappush
from typing import Literal, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray

from gridworld.core.world import WorldT
from gridworld.policy.ctf.typing import ObservationDict
from gridworld.typing import Position


class AStarNode(NamedTuple):
    f: int
    g: int
    h: int
    parent: Union["AStarNode", None]
    loc: Position


def a_star(
    start: Position,
    end: Position,
    field_map: NDArray[np.int_],
    world: WorldT,
    avoided_objects: list[str] = ["obstacle", "blue_agent", "red_agent"],
) -> list[Position]:
    """
    Compute the path from start to end using A* algorithm.

    Parameters
    ----------
    start : Position
        Start position
    end : Position
        End position
    field_map : NDArray[np.int_]
        Map of the environment
    world : WorldT
        World object

    Returns
    -------
    path: list[Position]
        List of positions from start to end
    """

    # Add the start and end nodes
    start_node = AStarNode(
        manhattan_distance(start, end), 0, manhattan_distance(start, end), None, start
    )
    # Initialize and heapify the lists
    open_nodes: list[AStarNode] = [start_node]
    closed_nodes: list[AStarNode] = []
    heapify(open_nodes)
    path: list[Position] = []  # return of the func

    while open_nodes:
        # Get the current node popped from the open list
        current_node = heappop(open_nodes)

        # Push the current node to the closed list
        closed_nodes.append(current_node)

        # When the goal is found
        if current_node.loc == end:
            current: AStarNode | None = current_node
            while current is not None:
                path.append(current.loc)
                current = current.parent

            path.reverse()
            break

        else:
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                # Get node location
                current_loc: Position = current_node.loc
                new_loc = (current_loc[0] + direction[0], current_loc[1] + direction[1])

                # Make sure within a range and avoid obstacles or other agents
                if loc_inside_map(new_loc, field_map) and loc_can_overlap(
                    new_loc, end, field_map, world, avoided_objects
                ):
                    # Create the f, g, and h values
                    g = current_node.g + 1
                    h = manhattan_distance(new_loc, end)
                    f = g + h

                    # Check if the new node is in the open or closed list
                    open_indices = [
                        i
                        for i, open_node in enumerate(open_nodes)
                        if open_node.loc == new_loc
                    ]
                    closed_indices = [
                        i
                        for i, closed_node in enumerate(closed_nodes)
                        if closed_node.loc == new_loc
                    ]

                    # Compare f values if the new node is already existing in either list
                    if closed_indices:
                        closed_index = closed_indices[0]
                        if f < closed_nodes[closed_index].f:
                            closed_nodes.pop(closed_index)
                            heappush(
                                open_nodes, AStarNode(f, g, h, current_node, new_loc)
                            )
                        else:
                            continue

                    elif open_indices:
                        open_index = open_indices[0]
                        if f < open_nodes[open_index].f:
                            open_nodes.pop(open_index)
                            open_nodes.append(AStarNode(f, g, h, current_node, new_loc))
                            heapify(open_nodes)
                        else:
                            continue

                    else:
                        heappush(open_nodes, AStarNode(f, g, h, current_node, new_loc))

                else:
                    continue

    if not path:
        raise ValueError("No path found")
    else:
        pass

    return path


def manhattan_distance(p1: Position, p2: Position) -> int:
    """
    Compute a Manhattan distance of two points

    Parameters
    ----------
    p1: Position
        Location
    p2 : Position
        Another location

    Returns
    -------
    distance : int
        Manhattan distance between two points
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def loc_inside_map(loc: Position, field_map: NDArray[np.int_]) -> bool:
    """
    Check if a location is inside the field_map

    Parameters
    ----------
    loc: Position
        Location
    field_map: NDArray[np.int_]
        Map of the environment

    Returns
    -------
    inside: bool
        True if the location is inside the field_map
    """
    rows, cols = field_map.shape
    x, y = loc
    return 0 <= x < rows and 0 <= y < cols


def loc_can_overlap(
    loc: Position,
    end_loc: Position,
    field_map: NDArray[np.int_],
    world: WorldT,
    avoided_objects: list[str],
) -> bool:
    """
    Check if a location can overlap with other objects.
    If the location is not empty and not in the list of avoided objects, it can overlap.
    In case the location is the end location, it can overlap regardless of the object at the location.


    Parameters
    ----------
    loc: Position
        Location
    end_loc: Position
        End location
    field_map: NDArray[np.int_]
        Map of the environment
    world: WorldT
        World object
    avoided_objects: list[str]
        List of objects to avoid

    Returns
    -------
    can_overlap: bool
        True if the location can overlap with other objects
    """
    x, y = loc
    can_overlap: bool = (
        True
        if loc == end_loc
        else world.IDX_TO_OBJECT[field_map[x][y]] not in avoided_objects
    )

    return can_overlap


def get_unterminated_opponent_pos(
    observation: ObservationDict, opponent_agent: Literal["red_agent", "blue_agent"]
) -> list[Position]:
    num_blue_agents: int = observation["blue_agent"].reshape([-1, 2]).shape[0]
    num_red_agents: int = observation["red_agent"].reshape([-1, 2]).shape[0]
    terminated_opponent_agents: NDArray[np.int_] = (
        observation["terminated_agents"][0:num_blue_agents]
        if opponent_agent == "blue_agent"
        else observation["terminated_agents"][
            num_blue_agents : num_blue_agents + num_red_agents
        ]
    )

    # Only choose the positions of the opponent agents that are not terminated
    opponent_pos_np: NDArray = observation[opponent_agent].reshape(-1, 2)[
        terminated_opponent_agents == 0
    ]
    opponent_pos: list[Position] = [tuple(pos) for pos in opponent_pos_np]

    return opponent_pos
