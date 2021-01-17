# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 100000000

        if successorGameState.isWin():
            return score
        elif successorGameState.isLose():
            return -score
        ghosts, ghostsDistances = [], []
        scaredGhosts, scaredGhostsDistances = [], []

        for ghost in successorGameState.getGhostPositions():
            if ghost:
                ghosts.append(ghost)
                ghostsDistances.append(manhattanDistance(newPos, ghost))
            else:
                scaredGhosts.append(ghost)
                scaredGhostsDistances.append(manhattanDistance(newPos, ghost.getPosition()))

        try:
            nearestGhost = min(ghostsDistances, default=0)
            ghostWeight = -10 / nearestGhost
        except ZeroDivisionError:
            ghostWeight = 1

        nearestScaredGhost = min(scaredGhostsDistances, default=0)

        foodList = newFood.asList()
        numOfFood = len(foodList)
        foodDistances = []
        for food in foodList:
            foodDistances.append(manhattanDistance(newPos, food))
        nearestFood = min(foodDistances, default=0)

        numberOfCapsulesLeft = len(currentGameState.getCapsules())
        score = ghostWeight + (- 5 * nearestFood) - (1000 * numOfFood) - numberOfCapsulesLeft - nearestScaredGhost
        return score

        # return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"

        # anazitisi_me_antipalotita page 19
        def maxValue(state, depth):
            # if TERMINAL-TEST(state) then return UTILITY(state)
            actions = state.getLegalActions(0)
            if actions == []:  # no more legal actions so the game is finished
                return self.evaluationFunction(state)
            if depth == self.depth:  # depth is the max depth
                return self.evaluationFunction(state)
            u = max(minValue(state.generateSuccessor(0, action), 1, depth + 1) for action in actions)
            return u

        def minValue(state, agentIndex, depth):
            actions = state.getLegalActions(agentIndex)
            numOfAgents = state.getNumAgents()
            if actions == []:
                return self.evaluationFunction(state)
            if agentIndex == numOfAgents - 1:  # all ghosts moved so now pacman has to move
                u = min(maxValue(state.generateSuccessor(agentIndex, action), depth) for action in actions)
            else:
                u = min(
                    minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in actions)
            return u

        maxNum = float('-inf')
        for x in gameState.getLegalActions(0):
            value = minValue(gameState.generateSuccessor(0, x), 1, 1)
            if value > maxNum:
                maxNum = value
                action = x

        return action
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #  anazitisi_me_antipalotita page 48
        def maxValue(state, depth, a, b):
            actions = state.getLegalActions(0)
            if actions == []:  # no more legal actions so the game is finished
                return self.evaluationFunction(state)
            if depth == self.depth:  # depth is the max depth
                return self.evaluationFunction(state)

            u = float('-inf')
            if depth == 0:
                selected = actions[0]
            for action in actions:
                nextU = minValue(state.generateSuccessor(0, action), 1, depth + 1, a, b)
                if nextU > u:
                    u = nextU
                    if depth == 0:
                        selected = action
                if u > b:
                    return u
                a = max(a, u)

            if depth == 0:
                return selected
            return u

        def minValue(state, agentIndex, depth, a, b):
            actions = state.getLegalActions(agentIndex)
            if actions == []:
                return self.evaluationFunction(state)

            u = float('inf')
            numOfActions = state.getNumAgents()
            for action in actions:
                if agentIndex == numOfActions - 1:
                    nextU = maxValue(state.generateSuccessor(agentIndex, action), depth, a, b)
                else:
                    nextU = minValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth, a, b)

                u = min(u, nextU)
                if u < a:
                    return u
                b = min(b, u)
            return u

        return maxValue(gameState, 0, float('-inf'), float('inf'))
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # adversarial search page 42
        def maxValue(state, depth):

            actions = state.getLegalActions(0)
            if actions == []:  # no more legal actions so the game is finished
                return self.evaluationFunction(state)
            if depth == self.depth:  # depth is the max depth
                return self.evaluationFunction(state)
            v = max(expValue(state.generateSuccessor(0, everyAction), 0 + 1, depth + 1) for everyAction in actions)
            return v

        def expValue(state, agentIndex, depth):
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            possibility = 1.0 / len(actions)
            u = 0
            for everyAction in actions:
                numOfActions = state.getNumAgents()
                if agentIndex == numOfActions - 1:
                    u += possibility * maxValue(state.generateSuccessor(agentIndex, everyAction), depth)
                else:
                    u += possibility * expValue(state.generateSuccessor(agentIndex, everyAction), agentIndex + 1, depth)
            return u

        maxNum = float('-inf')
        for x in gameState.getLegalActions():
            value = expValue(gameState.generateSuccessor(0, x), 1, 1)
            if value > maxNum:
                maxNum = value
                selectedAction = x
        return selectedAction
        # util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    if currentGameState.isWin():
        return 100000 + currentGameState.getScore()
    elif currentGameState.isLose():
        return -100000
    ghosts, ghostsDistances = [], []
    scaredGhosts, scaredGhostsDistances = [], []
    foodWeight, ghostWeight, scaredGhostWeight = 0, 0, 0
    allGhosts = currentGameState.getGhostStates()
    for ghost in allGhosts:
        if ghost.scaredTimer > 0:
            scaredGhosts.append(ghost)
        else:
            ghosts.append(ghost)
    for ghost in ghosts:
        ghostsDistances.append(manhattanDistance(pacmanPosition, ghost.getPosition()))
    for ghost in scaredGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition, ghost.getPosition()))
    for distance in ghostsDistances:
        if distance < 3:  # we actively try to avoid only the very close to the pacman ghosts
            ghostWeight += 10000 * distance
    for distance in scaredGhostsDistances:
        if distance: scaredGhostWeight += 10000 / distance
    foodList = currentGameState.getFood().asList()
    numOfFood = len(foodList)
    foodDistances = []
    for food in foodList:
        foodDistances.append(manhattanDistance(pacmanPosition, food))
    for foodDistance in foodDistances:
        if foodDistance: foodWeight += 500 / foodDistance
    numberOfCapsulesLeft = len(currentGameState.getCapsules())
    try:
        capsulesWeight = 1500 / numberOfCapsulesLeft
    except ZeroDivisionError:
        capsulesWeight = 1
    capsuleDistances = []
    capsuleDistanceWeight = 0
    for capsules in currentGameState.getCapsules():
        capsuleDistances.append(manhattanDistance(pacmanPosition, capsules))
    for capsuleDistance in capsuleDistances:
        if capsuleDistance: capsuleDistanceWeight += 100 / capsuleDistance
    scoreWeight = 100 * currentGameState.getScore()
    numOfFoodWeight = 100 / numOfFood
    result = scoreWeight + ghostWeight + foodWeight + numOfFoodWeight + capsulesWeight + scaredGhostWeight + capsuleDistanceWeight
    return result

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
