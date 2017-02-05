import numpy
import fake_robot as robot
from fractions import Fraction


# How do I make a GVF?
# Hmmm.


class Question(object):
    def __init__(self, policy, cumulant, continuation_gamma):
        self.policy = policy
        self.cumulant = cumulant
        self.continuation_gamma = continuation_gamma


class GeneralValueFunction():
    def __init__(self, num_features,
                 initial_on_features,
                 question,
                 initial_weights='random',
                 learning_rate=None,
                 trace_decay=0):
        # type: (int, numpy.array.Array, Question, numpy.array.Array, float, float) -> GeneralValueFunction

        self.features_new = None  # x(s'); a vector of features of the newest state
        self.cumulant = None  # Z; cumulant or pseudo-reward
        self.continuation_gamma = None  # gamma; the probability of continuation at this timestep

        self.delta = 0  # delta; difference between new pseudo-value and old pseudo-value

        self.eligibility_trace = numpy.zeros(num_features)  # e_t; a vector of ?? accumulating trace
        self.trace_decay = trace_decay  # lambda; the rate at which the eligibility_trace fades away
        if learning_rate is not None:
            self.learning_rate = learning_rate  # alpha;
        else:
            self.learning_rate = 0.01 / float(len(initial_on_features))

        # self.features_old  # x(s); a vector of features of the previous state
        self.features_old = initial_on_features

        # self.weights  # w; a vector of weights --- hmm?
        if initial_weights is 'random':
            # Set the initial w to small random values from -0.1 to +0.1
            self.weights = (0.2 * numpy.random.rand(num_features)) - 0.1
        else:
            self.weights = initial_weights

        self.question = question

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, sensor_reading, features_new, timestep):
        self.features_new = features_new
        self.cumulant = self.question.cumulant(sensor_reading, features_new, timestep)
        self.continuation_gamma = self.question.continuation_gamma(sensor_reading, features_new, timestep)


class GeneralValueFunctionTDLambda(GeneralValueFunction):
    def __init__(self, num_features, initial_on_features, question, initial_weights='random', trace_decay=0,
                 learning_rate=None):
        GeneralValueFunction.__init__(self, num_features, initial_on_features, question,
                                      initial_weights=initial_weights,
                                      trace_decay=trace_decay,
                                      learning_rate=learning_rate)

    def learn(self):
        self.delta = self.cumulant + self.continuation_gamma * sum(self.weights[self.features_new]) \
                     - sum(self.weights[self.features_old])

        self.eligibility_trace *= (self.trace_decay * self.continuation_gamma)
        self.eligibility_trace[self.features_old] += 1

        self.weights += ((self.learning_rate * self.delta) * self.eligibility_trace)
        # print(self.weights[[13, 21]])

        self.features_old = self.features_new

        # TODO: Remove the following, possibly. I thought this might reduce update bugs.
        self.features_new = None
        self.cumulant = None
        self.continuation_gamma = None


class GeneralValueFunctionGTD(GeneralValueFunction):
    def __init__(self):
        GeneralValueFunction.__init__(self, num_features, initial_on_features, question,
                                      initial_weights=initial_weights,
                                      trace_decay=trace_decay,
                                      learning_rate=learning_rate)

        # Used for off-policy predictions, so we need a policy.
        assert (self.question.policy is not None)

        self.importance_ratio = None  # rho; probability of action under target policy divided by prob under behaviour policy

    def learn(self, action):
        pass


class PolicyTracker(object):
    # Yay for fractions!

    def __init__(self):
        self.policy_probs = {}
        self.policy_views = {}

    def add_to_policy(self, state, action):
        if state not in self.policy_probs:
            self.policy_probs[state] = {action: Fraction(1)}
            self.policy_views[state] = 1
        else:
            self.policy_views[state] += 1
            k = self.policy_views[state]

            for a in self.policy_probs[state]:
                self.policy_probs[state][a] *= Fraction(k - 1, k)
            if action in self.policy_probs[state]:
                self.policy_probs[state][action] += Fraction(1, k)
            else:
                self.policy_probs[state][action] = Fraction(1, k)

    def get_probability(self, state, action):
        if state not in self.policy_probs or action not in self.policy_probs[state]:
            return 0
        else:
            return self.policy_probs[state][action]


if __name__ == '__main__':
    pass
