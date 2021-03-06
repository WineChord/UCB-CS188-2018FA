# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            state_values = util.Counter()
            for state in self.mdp.getStates():
                action_values = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    action_values[action] = self.getQValue(state, action)
                state_values[state] = action_values[action_values.argMax()]
            for state in self.mdp.getStates():
                self.values[state] = state_values[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        q_value = 0
        nextState_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in nextState_prob:
            q_value += prob * (self.mdp.getReward(state, action, nextState)+self.discount*self.values[nextState])
        return q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return None
        actions  = self.mdp.getPossibleActions(state)
        max_q = self.getQValue(state, actions[0])
        opt_a = actions[0]
        for i in range(1, len(actions)):
            next_q = self.getQValue(state, actions[i])
            if next_q > max_q:
                max_q = next_q
                opt_a = actions[i]
        return opt_a

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        state_size = len(states)
        for i in range(self.iterations):
            state = states[i%state_size]
            action_values = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                action_values[action] = self.getQValue(state, action)
            self.values[state] = action_values[action_values.argMax()]


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for s in self.mdp.getStates():
            predecessors[s] = set()
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for next_s, _ in self.mdp.getTransitionStatesAndProbs(s,a):
                    predecessors[next_s].add(s)
        pq = util.PriorityQueue()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            action_values = util.Counter()
            for action in self.mdp.getPossibleActions(s):
                action_values[action] = self.getQValue(s, action)
            max_q = action_values[action_values.argMax()]
            diff = abs(self.values[s]-max_q)
            pq.update(s, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            action_values = util.Counter()
            for action in self.mdp.getPossibleActions(s):
                action_values[action] = self.getQValue(s, action)
            self.values[s] = action_values[action_values.argMax()]
            for p in predecessors[s]:
                p_values = util.Counter()
                for action in self.mdp.getPossibleActions(p):
                    p_values[action] = self.getQValue(p, action)
                max_q = p_values[p_values.argMax()]
                diff = abs(self.values[p]-max_q)
                if diff > self.theta:
                    pq.update(p, -diff)