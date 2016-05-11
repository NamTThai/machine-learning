import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        # qLearn is a HashTable where key is a state and value is another HashTable
        # this sub-HashTable has key is an action and value is Q(s,a)
        # sample structure:
        # {
        #     s1: {a1: Q(s1, a1), a2: Q(s1, a2)},
        #     s2: {a1: Q(s2, a1), a2: Q(s2, a2)}
        # }
        self.qLearn = {}

        # reward discount rate is between [0,1]
        self.gamma = 0.1

        # testingTheWater is a variable to overcome local minimum
        self.testingTheWater = False

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        self.state = State.checkForNewState(self.next_waypoint, inputs['light'])

        # TODO: Select action according to your policy
        # First choice is random before using Q values
        action = random.choice(self.env.valid_actions)
        if not self.testingTheWater or random.randint(0,1):
            if self.state in self.qLearn.keys():
                maxQ = 0
                for storedAction in self.qLearn[self.state].iterkeys():
                    storedQValue = self.qLearn[self.state][storedAction]
                    if storedQValue > maxQ:
                        action = storedAction
                        maxQ = storedQValue

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if self.state not in self.qLearn.keys():
            self.qLearn[self.state] = {}
        qVal = reward
        # Calculate discounted reward of future states
        for state in State.getPossibleNextStates(self.qLearn, self.state, action):
            actionDict = self.qLearn[state]
            # Compute max Q(s', a')
            futureQ = 0
            for storedAction in actionDict.iterkeys():
                futureQ = max(futureQ, actionDict[storedAction])
            qVal += self.gamma * futureQ
        # Store new Q value in the qLearn dictionary
        self.qLearn[self.state][action] = qVal

        # print action
        # print self.state
        # print reward
        # print("Updating")
        # for state in self.qLearn.iterkeys():
        #     print(state.light, state.next_waypoint, self.qLearn[state])
        # print "Updated"

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
    a.testingTheWater = True
    sim.run(n_trials=50)  # press Esc or close pygame window to quit
    a.testingTheWater = False
    sim.run(n_trials=50)

class State(object):
    """A wrapper class to model Agent's states"""
    # There should be a maximum of 6 states (2 possible lights and 3 possible next_waypoint)
    AVAILABLE_STATES = []

    def __init__(self, next_waypoint, light):
        self.next_waypoint = next_waypoint
        self.light = light

    def __str__(self):
        return "(next_waypoint: {}, light: {})".format(self.next_waypoint, self.light)

    @staticmethod
    def checkForNewState(next_waypoint, light):
        # Check if this state already existed
        for state in State.AVAILABLE_STATES:
            if state.next_waypoint == next_waypoint and state.light == light:
                return state
        # initialize a new state variable if not existed
        state = State(next_waypoint, light)
        State.AVAILABLE_STATES.append(state)
        return state

    @staticmethod
    def getPossibleNextStates(stateMap, state, action):
        """ This method returns all s' where T(s, a, s') = Pr(s'|s, a) = 1 """
        if action is not None:
            return stateMap.keys()

        # When you don't do anything, next_waypoint cannot change
        possibleNextStates = []
        for storedState in stateMap.iterkeys():
            if state.next_waypoint == storedState.next_waypoint:
                possibleNextStates.append(storedState)
        return possibleNextStates


if __name__ == '__main__':
    run()
