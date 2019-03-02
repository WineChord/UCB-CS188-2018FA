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

def pS():
    print('*' * 20)

def mDis(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x2-x1) + abs(y2-y1)

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
        # print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # return Directions.STOP
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
        oldFood = currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        myscore = 0
        if oldFood[newPos[0]][newPos[1]]:
            myscore += 2

        # print("myscore after food:"+ str(myscore))
        for i, row in enumerate(newFood):
            for j, val in enumerate(row):
                if val:
                    if (i,j) == newPos:
                        continue
                    else:
                        myscore += 1/mDis((i,j), newPos) * 1.5
        for i in newScaredTimes:
            myscore += i/10
        for g in newGhostStates:
            gPos = g.getPosition()
            if gPos == newPos:
                myscore -= 10
            else:
                myscore -= 1/mDis(gPos, newPos) * 12
        return myscore
        "*** YOUR CODE HERE ***"
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def value(self, gameState, depth, agentIndex):
        if depth == -1 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex % gameState.getNumAgents() == 0:  # max agent
            return self.maxValue(gameState, depth, 0)
        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        v = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.value(successor, depth, agentIndex+1))
        return v

    def minValue(self, gameState, depth, agentIndex):
        v = float('inf')
        i = agentIndex
        legal_actions = gameState.getLegalActions(i)
        for action in legal_actions:
            successor = gameState.generateSuccessor(i, action)
            if i == gameState.getNumAgents()-1:
                v = min(v, self.value(successor, depth-1, agentIndex+1))
            else:
                v = min(v, self.value(successor, depth, agentIndex+1))
        return v

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
        # util.raiseNotDefined()
        legal_actions = gameState.getLegalActions(0)
        scores = [self.value(gameState.generateSuccessor(0, action), self.depth-1, 1) for action in legal_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legal_actions[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, gameState, depth, agentIndex, alpha, beta):
        if depth == -1 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex % gameState.getNumAgents() == 0:  # max agent
            return self.maxValue(gameState, depth, 0, alpha, beta)
        else:
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        v = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.value(successor, depth, agentIndex+1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        v = float('inf')
        i = agentIndex
        legal_actions = gameState.getLegalActions(i)
        for action in legal_actions:
            successor = gameState.generateSuccessor(i, action)
            if i == gameState.getNumAgents()-1:
                v = min(v, self.value(successor, depth-1, agentIndex+1, alpha, beta))
            else:
                v = min(v, self.value(successor, depth, agentIndex+1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        res = Directions.STOP
        alpha = float('-inf')
        beta = float('inf')
        v = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            succv = self.value(successor, self.depth-1, 1, alpha, beta)
            # print("succv", succv)
            if succv > v:
                v = succv
                res = action
            if v > beta:
                res = action
                break
            alpha = max(alpha, v)
        # print("v", v)
        return res
        # scores = [self.value(gameState.generateSuccessor(0, action), self.depth - 1, 1, alpha, beta) for action in legal_actions]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        # return legal_actions[chosenIndex]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self, gameState, depth, agentIndex):
        if depth == -1 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex % gameState.getNumAgents() == 0:  # max agent
            return self.maxValue(gameState, depth, 0)
        else:
            return self.expValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        v = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.value(successor, depth, agentIndex+1))
        return v

    def expValue(self, gameState, depth, agentIndex):
        v = float('inf')
        i = agentIndex
        legal_actions = gameState.getLegalActions(i)
        num_actions = len(legal_actions)
        p = 1/num_actions
        v = 0
        for action in legal_actions:
            successor = gameState.generateSuccessor(i, action)
            if i == gameState.getNumAgents() - 1:
                # v = min(v, self.value(successor, depth - 1, agentIndex + 1))
                v += p * self.value(successor, depth - 1, agentIndex + 1)
            else:
                # v = min(v, self.value(successor, depth, agentIndex + 1))
                v += p * self.value(successor, depth, agentIndex + 1)
        return v

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # print("entering expectimax.getAction...")
        legal_actions = gameState.getLegalActions(0)
        scores = [self.value(gameState.generateSuccessor(0, action), self.depth - 1, 1) for action in legal_actions]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legal_actions[chosenIndex]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # return currentGameState.getScore()
    foodPos = currentGameState.getFood()
    pacPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    myscore = 0
    if foodPos[pacPos[0]][pacPos[1]]:
        myscore += 2

    for i, row in enumerate(foodPos):
        for j, val in enumerate(row):
            if val:
                if (i, j) == pacPos:
                    myscore += 3
                else:
                    myscore += 1 / mDis((i, j), pacPos) * 3

    for i in newScaredTimes:
        myscore += i / 10
    for g in newGhostStates:
        gPos = g.getPosition()
        if gPos == pacPos:
            myscore -= 10
        # else:
            # myscore -= 1 / mDis(gPos, pacPos) * 12
    return myscore + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
