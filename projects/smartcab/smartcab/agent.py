import random
import exceptions
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # qLearn is a HashTable where key is a state and value is another HashTable
        # this sub-HashTable has key is an action and value is Q(s,a)
        # sample structure:
        # {
        #     s1: {a1: Q(s1, a1), a2: Q(s1, a2)},
        #     s2: {a1: Q(s2, a1), a2: Q(s2, a2)}
        # }
        self.qLearn = {}
        for state in State.AVAILABLE_STATES:
            self.qLearn[state] = {}
            for action in Environment.valid_actions:
                # Initialize all Q values to be 0
                self.qLearn[state][action] = 0

        # reward discount rate is between [0,1]
        # TODO: how to choose the best gamma
        self.gamma = 0.1

        # learning rate is between [0,1]
        # TODO: how to choose the best starting learning_rate
        self.learning_rate = 1

        # Exploration rate is for simulated annealing approach for action selection
        # TODO: how to choose the best exploration_rate
        self.exploration_rate = 1

        # Use to decay learning_rate and exploration_rate
        # TODO: how to decay adequately (not too fast, not too slow)
        self.decay_denominator = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Decaying rate at the end of each iteration
        self.learning_rate = self.decay_rate(self.learning_rate)
        self.exploration_rate = self.decay_rate(self.exploration_rate)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = State.getState(self.next_waypoint, inputs['light'])

        # Select action according to your policy
        # Decide whether to select randomly or select based on Q table
        action = random.choice(self.env.valid_actions)
        if random.random() > self.exploration_rate:
            # Check if there is an existing Q(s,a) mapping
            # If not, just use the randomly picked action
            if self.state in self.qLearn.keys():
                maxQ = 0
                for storedAction in self.qLearn[self.state].iterkeys():
                    storedQValue = self.qLearn[self.state][storedAction]
                    if storedQValue > maxQ:
                        action = storedAction
                        maxQ = storedQValue

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        qVal = reward
        # Calculate discounted reward of future states
        for state in self.qLearn.iterkeys():
            actionDict = self.qLearn[state]
            # Compute max Q(s', a')
            qPrime = 0
            for storedAction in actionDict.iterkeys():
                qPrime = max(qPrime, actionDict[storedAction])
            qVal += self.gamma * qPrime
        # Calculate with learning rate
        qValOld = self.qLearn[self.state][action]
        qVal = (1 - self.learning_rate) * qValOld + self.learning_rate * qVal
        # Store new Q value in the qLearn dictionary
        self.qLearn[self.state][action] = qVal

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def decay_rate(self, rate):
        self.decay_denominator += 1
        # decay rate increases every 15 trials
        if self.decay_denominator % 15 == 0:
            return rate / float(self.decay_denominator * self.decay_denominator)
        else:
            return rate


class State(object):
    """A wrapper class to model Agent's states because tuple is not iterable
    and hence cannot be used as key for python dictionary
    """
    # There should be a maximum of 6 states (2 possible lights and 3 possible next_waypoint)
    AVAILABLE_STATES = []

    def __init__(self, next_waypoint, light):
        self.next_waypoint = next_waypoint
        self.light = light

    def __str__(self):
        return "(next_waypoint: {}, light: {})".format(self.next_waypoint, self.light)

    @staticmethod
    def getState(next_waypoint, light):
        for state in State.AVAILABLE_STATES:
            if state.next_waypoint == next_waypoint and state.light == light:
                return state
        # State not found
        raise exceptions.ValueError('State not found: next_waypoint {}, light {}'.format(next_waypoint, light))

    @staticmethod
    def initializeAvailableStates():
        for light in ['red', 'green']:
            for action in Environment.valid_actions[1:]:
                state = State(action, light)
                State.AVAILABLE_STATES.append(state)


def run():
    """Run the agent for a finite number of trials."""
    # Set up Q Learning
    State.initializeAvailableStates()

    # How many time to run this simulation
    num_sims = 1
    for n in xrange(num_sims):
        print "Trial {}".format(n)

        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

        # Now simulate it
        sim = Simulator(e, update_delay=0.1)  # reduce update_delay to speed up simulation
        sim.run(n_trials=100)  # press Esc or close pygame window to quit

        # Debug: After the simulation run, check if the result is as expected
        correct = True
        for state in a.qLearn.iterkeys():
            print(state.light, state.next_waypoint, a.qLearn[state])
            qMax = 0
            for action in a.qLearn[state]:
                qMax = max(qMax, a.qLearn[state][action])
                if state == State.getState('left', 'red') and action is None and qMax is not a.qLearn[state][action]:
                    correct = False
                if state == State.getState('forward', 'red') and action is None and qMax is not a.qLearn[state][action]:
                    correct = False
                if state == State.getState('right', 'red') and action == 'right' and qMax is not a.qLearn[state][action]:
                    correct = False
                if state == State.getState('left', 'green') and action == 'left' and qMax is not a.qLearn[state][action]:
                    correct = False
                if state == State.getState('forward', 'green') and action == 'forward' and qMax is not a.qLearn[state][action]:
                    correct = False
                if state == State.getState('right', 'green') and action == 'right' and qMax is not a.qLearn[state][action]:
                    correct = False
        if not correct:
            print("Agent has not achived optimal strategy")


if __name__ == '__main__':
    run()
