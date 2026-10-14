"""
In this file, you will implement code relating to simple single-agent searches.
"""

import random
import typing

import pacai.agents.searchproblem
import pacai.core.action
import pacai.core.agent
import pacai.core.agentaction
import pacai.core.board
import pacai.core.gamestate
import pacai.core.search
import pacai.pacman.board
import pacai.search.common
import pacai.search.food
import pacai.search.position
import pacai.util.containers

def depth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements depth first search (DFS).
    This means that it will search the deepest nodes in the search tree first.
    See: https://en.wikipedia.org/wiki/Depth-first_search .
    """

    # *** Your Code Here ***
    start = problem.get_starting_node()
    frontier: pacai.util.containers.Stack[tuple[pacai.core.search.SearchNode, list[pacai.core.action.Action], float]] = pacai.util.containers.Stack()
    explored = set()

    frontier.push((start, [], 0.0))

    while (not frontier.is_empty()):
        node, actions, cost = frontier.pop()

        if node in explored:
            continue

        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(
                actions, cost, node
            )

        explored.add(node)

        for successor in problem.get_successor_nodes(node):
            next_node = successor.node

            if next_node not in explored:
                new_actions = actions + [successor.action]
                new_cost = cost + successor.cost

                frontier.push((next_node, new_actions, new_cost))

    raise pacai.core.search.SolutionNotFoundError()

def breadth_first_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements breadth first search (BFS).
    This means that it will search nodes based on what level in search tree they appear.
    See: https://en.wikipedia.org/wiki/Breadth-first_search .
    """

    # *** Your Code Here ***

    start = problem.get_starting_node()
    frontier: pacai.util.containers.Queue[tuple[pacai.core.search.SearchNode, list[pacai.core.action.Action], float]] = pacai.util.containers.Queue()
    explored = set()

    frontier.push((start, [], 0.0))

    while (not frontier.is_empty()):
        node, actions, cost = frontier.pop()

        if node in explored:
            continue

        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(
                actions, cost, node
            )

        explored.add(node)

        for successor in problem.get_successor_nodes(node):
            next_node = successor.node

            if next_node not in explored:
                new_actions = actions + [successor.action]
                new_cost = cost + successor.cost

                frontier.push((next_node, new_actions, new_cost))

    raise pacai.core.search.SolutionNotFoundError()

def uniform_cost_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements uniform cost search (UCS).
    This means that it will search nodes with a lower total cost first.
    See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Practical_optimizations_and_infinite_graphs .
    """

    # *** Your Code Here ***
    start = problem.get_starting_node()

    frontier: pacai.util.containers.PriorityQueue[
        tuple[pacai.core.search.SearchNode, list[pacai.core.action.Action], float]
    ] = pacai.util.containers.PriorityQueue()
    explored = set()

    frontier.push((start, [], 0.0), 0.0)

    while not frontier.is_empty():
        node, actions, cost = frontier.pop()

        if node in explored:
            continue

        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(
                actions, cost, node
            )

        explored.add(node)

        for successor in problem.get_successor_nodes(node):
            next_node = successor.node

            if next_node not in explored:
                new_actions = actions + [successor.action]
                new_cost = cost + successor.cost

                frontier.push(
                    (next_node, new_actions, new_cost),
                    new_cost
                )

    raise pacai.core.search.SolutionNotFoundError()


def astar_search(
        problem: pacai.core.search.SearchProblem,
        heuristic: pacai.core.search.SearchHeuristic,
        rng: random.Random,
        **kwargs: typing.Any) -> pacai.core.search.SearchSolution:
    """
    A pacai.core.search.SearchProblemSolver that implements A* search (pronounced "A Star search").
    This means that it will search nodes with a lower combined cost and heuristic first.
    See: https://en.wikipedia.org/wiki/A*_search_algorithm .
    """

    # *** Your Code Here ***
    start = problem.get_starting_node()

    frontier: pacai.util.containers.PriorityQueue[
        tuple[pacai.core.search.SearchNode, list[pacai.core.action.Action], float]
    ] = pacai.util.containers.PriorityQueue()
    explored = set()

    frontier.push((start, [], 0.0), heuristic(start, problem))

    while not frontier.is_empty():
        node, actions, cost = frontier.pop()

        if node in explored:
            continue

        if problem.is_goal_node(node):
            return pacai.core.search.SearchSolution(actions, cost, node)

        explored.add(node)

        for successor in problem.get_successor_nodes(node):
            next_node = successor.node

            if next_node not in explored:
                new_actions = actions + [successor.action]
                new_cost = cost + successor.cost

                priority = new_cost + heuristic(next_node, problem)

                frontier.push(
                    (next_node, new_actions, new_cost), priority)

    raise pacai.core.search.SolutionNotFoundError()

class CornersSearchNode(pacai.core.search.SearchNode):
    """
    A search node the can be used to represent the corners search problem.

    You get to implement this search node however you want.
    """
    def __init__(self,
            position: pacai.core.board.Position,
            visited_corners: typing.Iterable[pacai.core.board.Position] | None = None,
            ) -> None:
        self.position: pacai.core.board.Position = position
        self.visited_corners: frozenset[pacai.core.board.Position] = frozenset(
            visited_corners if visited_corners is not None else ())

    def __eq__(self, other: object) -> bool:
        if (not isinstance(other, CornersSearchNode)):
            return False

        return ((self.position == other.position)
                and (self.visited_corners == other.visited_corners))

    def __hash__(self) -> int:
        return hash((self.position, self.visited_corners))

    def __lt__(self, other: object) -> bool:
        if (not isinstance(other, CornersSearchNode)):
            return False

        return ((self.position, tuple(sorted(self.visited_corners)))
                < (other.position, tuple(sorted(other.visited_corners))))

class CornersSearchProblem(pacai.core.search.SearchProblem[CornersSearchNode]):
    """
    A search problem for touching the four different corners in a board.

    You may assume that very board is surrounded by walls (e.g., (0, 0) is a wall),
    and that the position diagonally inside from the walled corner is the location we are looking for.
    For example, if we had a square board that was 10x10, then we would be looking for the following corners:
     - (1, 1) -- North-West / Upper Left
     - (1, 8) -- North-East / Upper Right
     - (8, 1) -- South-West / Lower Left
     - (8, 8) -- South-East / Lower Right
    """
    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)

        self.board: pacai.core.board.Board = game_state.board
        start_position = game_state.get_agent_position()

        if start_position is None:
            raise ValueError("Could not find starting position.")

        self.start_position: pacai.core.board.Position = start_position
        self.corners: frozenset[pacai.core.board.Position] = frozenset(
            self.board.get_corners(offset=1))

    def get_starting_node(self) -> CornersSearchNode:
        # *** Your Code Here ***
        visited: frozenset[pacai.core.board.Position] = frozenset()

        if self.start_position in self.corners:
            visited = frozenset({self.start_position})

        return CornersSearchNode(self.start_position, visited)

    def is_goal_node(self, node: CornersSearchNode) -> bool:
        # *** Your Code Here ***
        return node.visited_corners == self.corners

    def get_successor_nodes(self, node: CornersSearchNode) -> list[pacai.core.search.SuccessorInfo]:
        successors = []

        for action, next_position in self.board.get_neighbors(node.position):
            visited = node.visited_corners

            if next_position in self.corners:
                visited = visited | {next_position}

            next_node = CornersSearchNode(next_position, visited)

            successors.append(pacai.core.search.SuccessorInfo(next_node, action, 1.0))

        self.expanded_node_count += 1

        if node not in self.visited_nodes:
            self.position_history.append(node.position)
            self.visited_nodes.add(node)

        return successors

def corners_heuristic(node: CornersSearchNode, problem: CornersSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for CornersSearchProblem.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # *** Your Code Here ***
    return pacai.search.common.null_heuristic(node, problem)  # Default to a trivial solution.

def food_heuristic(node: pacai.search.food.FoodSearchNode, problem: pacai.search.food.FoodSearchProblem, **kwargs: typing.Any) -> float:
    """
    A heuristic for the FoodSearchProblem.
    """

    # *** Your Code Here ***
    return pacai.search.common.null_heuristic(node, problem)  # Default to a trivial solution.

class ClosestDotSearchAgent(pacai.agents.searchproblem.GreedySubproblemSearchAgent):
    """
    Search for a path to all the food by greedily searching for the next closest food again and again
    (util we have reached all the food).

    This agent is left to you to fill out.
    But make sure to take your time and think.
    The final solution is quite simple if you take your time to understand everything up until this point and leverage
    pacai.agents.searchproblem.GreedySubproblemSearchAgent and pacai.student.problem.AnyMarkerSearchProblem.
    pacai.agents.searchproblem.GreedySubproblemSearchAgent is already implemented,
    but you should take some time to understand it.
    pacai.student.problem.AnyMarkerSearchProblem (below in this file) has not yet been implemented,
    but is the quickest and easiest way to implement this class.

    Hint:
    Remember that you can call a parent class' `__init__()` method from a child class' `__init__()` method.
    (See pacai.student.problem.AnyMarkerSearchProblem for an example.)
    Child classes will generally always call their parent's `__init__()` method
    (if a child class does not implement `__init__()`, then the parent's `__init__()` is automatically called).
    This call does not need to be the first line in the method,
    and you can pass whatever you want to the parent's `__init__()`.
    """

    # *** Your Code Here ***

class AnyMarkerSearchProblem(pacai.search.position.PositionSearchProblem):
    """
    A search problem for finding a path to any instance of the specified board marker (e.g., food, wall, power capsule).

    This search problem is just like the pacai.search.position.PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    You may modify the `__init__()` if you want, the other methods should be fine as-is.
    """

    def __init__(self,
            game_state: pacai.core.gamestate.GameState,
            target_marker: pacai.core.board.Marker = pacai.pacman.board.MARKER_PELLET,
            **kwargs: typing.Any) -> None:
        super().__init__(game_state, **kwargs)

        # *** Your Code Here ***

    def is_goal_node(self, node: pacai.search.position.PositionSearchNode) -> bool:
        # *** Your Code Here ***
        raise NotImplementedError('CornersSearchProblem.is_goal_node')

class ApproximateSearchAgent(pacai.core.agent.Agent):
    """
    A search agent that tries to perform an approximate search instead of an exact one.
    In other words, this agent is okay with a solution that is "good enough" and not necessarily optimal.
    """
