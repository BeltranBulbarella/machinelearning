# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from util import Stack
from util import Queue
from util import PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
        Returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        if actions is None:
            return 999999
        return len(actions)


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    frontier = Stack()
    startState = problem.getStartState()
    frontier.push((startState, [], set()))

    while not frontier.isEmpty():
        currentState, actions, visited = frontier.pop()

        if problem.isGoalState(currentState):
            return actions

        visited.add(currentState)

        for nextState, action, cost in problem.getSuccessors(currentState):
            if nextState not in visited:
                newActions = actions + [action]
                newVisited = visited.copy()
                frontier.push((nextState, newActions, newVisited))

    return []


def breadthFirstSearch(problem):
    frontier = Queue()
    startState = problem.getStartState()
    frontier.push((startState, []))

    explored = set()

    while not frontier.isEmpty():
        currentState, actions = frontier.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in explored:
            explored.add(currentState)

            for nextState, action, cost in problem.getSuccessors(currentState):
                if nextState not in explored:
                    newActions = actions + [action]
                    frontier.push((nextState, newActions))

    return []

def uniformCostSearch(problem):
    frontier = PriorityQueue()
    startState = problem.getStartState()
    frontier.push((startState, [], 0), 0)

    visited = {}

    while not frontier.isEmpty():
        currentState, actions, currentCost = frontier.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited or visited[currentState] > currentCost:
            visited[currentState] = currentCost

            # Process each successor
            for nextState, action, cost in problem.getSuccessors(currentState):
                newCost = currentCost + cost
                newActions = actions + [action]
                if nextState not in visited or visited[nextState] > newCost:
                    frontier.update((nextState, newActions, newCost), newCost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    frontier = PriorityQueue()
    startState = problem.getStartState()
    startHeuristic = heuristic(startState, problem)
    frontier.push((startState, [], 0), startHeuristic)

    visited = {}

    while not frontier.isEmpty():
        currentState, actions, currentCost = frontier.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited or visited[currentState] > currentCost:
            visited[currentState] = currentCost

            for nextState, action, cost in problem.getSuccessors(currentState):
                newCost = currentCost + cost
                newHeuristic = heuristic(nextState, problem)
                totalCost = newCost + newHeuristic
                newActions = actions + [action]
                if nextState not in visited or visited[nextState] > newCost:
                    frontier.update((nextState, newActions, newCost), totalCost)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
