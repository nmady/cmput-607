import numpy
import fake_robot as robot
from fractions import Fraction


# How do I make a GVF?
# Hmmm.


class Question(object):
    def __init__(self, policy_keeper, cumulant, continuation_gamma):
        self.policy_keeper = policy_keeper
        self.cumulant = cumulant # is a function which takes sensor_reading, on_features, and timestep
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

        self.set_eligibility_trace()

        self.weights += ((self.learning_rate * self.delta) * self.eligibility_trace)

        self.features_old = self.features_new

        # TODO: Remove the following, possibly. I thought this might reduce update bugs.
        self.features_new = None
        self.cumulant = None
        self.continuation_gamma = None

    def set_eligibility_trace(self):
        self.eligibility_trace *= (self.trace_decay * self.continuation_gamma)
        self.eligibility_trace[self.features_old] += 1


class GeneralValueFunctionGTD(GeneralValueFunctionTDLambda):
    def __init__(self, num_features, initial_on_features, question,
                                      initial_weights='random',
                                      trace_decay=0,
                                      learning_rate=None):
        GeneralValueFunctionTDLambda.__init__(self, num_features, initial_on_features, question,
                                      initial_weights=initial_weights,
                                      trace_decay=trace_decay,
                                      learning_rate=learning_rate)

        # Used for off-policy predictions, so we need a policy.
        assert (self.question.policy_keeper is not None)

        self.importance_ratio = None  # rho; probability of action under target policy divided by prob under behaviour policy

    def update(self, sensor_reading, features_new, timestep, action, prob_under_behaviour):
        GeneralValueFunction.update(self, sensor_reading, features_new, timestep)

        #TODO: Should the state I'm passing be the sensor_reading or the feature representation of it (features_new)?
        prob_under_target = self.question.policy_keeper.get_probability(sensor_reading, action)
        self.importance_ratio = Fraction(prob_under_target)/Fraction(prob_under_behaviour)

    def learn(self):
        GeneralValueFunctionTDLambda.learn(self)
        self.importance_ratio = None

    def set_eligibility_trace(self):
        self.eligibility_trace *= (self.trace_decay * self.continuation_gamma)
        self.eligibility_trace[self.features_old] += 1
        self.eligibility_trace *= float(self.importance_ratio)


class PolicyKeeper(object):
    # Yay for fractions!

    def __init__(self, policy_probs=None, policy_views=None, policy_func=None):
        self.policy_func = policy_func
        if policy_probs:
            assert(type(policy_probs) == dict)
            self.policy_probs = policy_probs
        else:
            self.policy_probs = {}

        if policy_views:
            assert(type(policy_views) == dict)
            self.policy_views = policy_views
        else:
            self.policy_views = {}

    def add_to_policy(self, state, action):
        if state not in self.policy_probs:
            self.policy_probs[state] = {action: Fraction(0)}

        if state not in self.policy_views:
            self.policy_views[state] = 0

        self.policy_views[state] += 1
        k = self.policy_views[state]

        for a in self.policy_probs[state]:
            self.policy_probs[state][a] *= Fraction(k - 1, k)
        if action in self.policy_probs[state]:
            self.policy_probs[state][action] += Fraction(1, k)
        else:
            self.policy_probs[state][action] = Fraction(1, k)

    def get_probability(self, state, action):
        if self.policy_func:
            # then we know the policy perfectly and can return the probability for any pair
            return self.policy_func(state, action)

        # Otherwise, we are approximating the policy and return a guess from our table.
        elif state not in self.policy_probs or action not in self.policy_probs[state]:
            return 0
        else:
            return self.policy_probs[state][action]


class Verifier(object):

    def __init__(self, gvf, initial_timestep, wait_time=100):
        """

        :type gvf: GeneralValueFunction
        """
        self.gvf = gvf
        self.initial_timestep = initial_timestep
        self.wait_time = wait_time

        self.last_timestep = None
        self.timestep_return_is_for = initial_timestep

        self.gammas = []
        self.cumulants = []

    def update(self, timestep):
        assert(timestep != self.last_timestep)
        assert(gvf.continuation_gamma is not None)
        assert(gvf.cumulant is not None)
        self.last_timestep = timestep
        self.gammas.append(gvf.continuation_gamma)
        self.cumulants.append(gvf.cumulant)
        if len(self.gammas) > self.wait_time:
            self.timestep_return_is_for += 1
            self.gammas.pop(0)
            self.cumulants.pop(0)

    def get_idealized_return(self):
        if len(self.gammas) < self.wait_time:
            return None
        else:
            # idealized_return = 0
            # for k in range(self.wait_time):
            #    idealized_return += numpy.product(self.gammas[:k]) * self.cumulant[k]

            return self.timestep_return_is_for, sum([numpy.product(self.gammas[:k]) * self.cumulants[k] for k in range(self.wait_time)])




if __name__ == '__main__':

    gvf = GeneralValueFunctionTDLambda(num_features=1, initial_on_features=[0], question=None)

    test_verifier = Verifier(gvf, 0, wait_time=30)

    for t in range(31):
        gvf.cumulant = 1
        gvf.continuation_gamma = 0.8
        test_verifier.update(t)

        print('self.gammas: ' + str(test_verifier.gammas))
        print('self.cumulants: ' + str(test_verifier.cumulants))
        print('idealized return: ' + str(test_verifier.get_idealized_return()))




