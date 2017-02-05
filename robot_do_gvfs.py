import fake_robot as robot
import general_value_functions as gvfs
import simple_tiles
import simple_kanerva
import sys, signal
import random
from fractions import Fraction

import numpy


random.seed(1)


def random_walk_policy(sensor_readings, on_features):
    return random.choice([412, 512, 612])


if __name__ == '__main__':

    def signal_handler(signal, frame):
        # http://stackoverflow.com/questions/18994912/ending-an-infinite-while-loop
        print("\n Program exiting gracefully")
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    myRobot = robot.Robot()

    t = 0

    sensor_readings = myRobot.get_readings()
    corners = myRobot.get_sensorspace_corners()

    # Here we set our behaviour policy!
    behaviour_policy = random_walk_policy
    behaviour_policy_keeper = gvfs.PolicyKeeper(policy_func=(lambda s,a: Fraction(1,3) if a in [412, 512, 612] else Fraction(0)))

    # Here we set up our initial feature vector, x!

    if len(sys.argv) > 1 and sys.argv[1] == 'tilecoding':
        num_bins = 10
        num_features = simple_tiles.get_num_features(num_bins, len(sensor_readings))
        on_features = simple_tiles.get_x(sensor_readings, corners[0], corners[1], num_bins)
        simple_tiles.show_bins(corners[0], corners[1], num_bins)
    else:
        num_on_features = 15
        num_features = 50
        coding = simple_kanerva.KanervaCoding(corners[0], corners[1], num_features, random_seed=10)
        on_features = coding.get_x(sensor_readings, num_on_features, distance_metric='hamming')

    # Here we set up our demons!

    # meant to ask how many steps until the temp is 40
    question1 = gvfs.Question(None, cumulant=(lambda x,y,z: 0 if x[2] == 40 else 1), continuation_gamma=(lambda x,y,z: 0 if x[2] == 40 else 1))

    # meant to ask how many steps until the position is > 550 if we were to always pick 612
    # policy2 = gvfs.PolicyKeeper(policy_probs={[0,0,412]:{612:Fraction(1)},[0,0,512]:{612:Fraction(1)},[0,0,612]:{612:Fraction(1)}})
    policy2 = gvfs.PolicyKeeper(policy_func=(lambda s,a: Fraction(1) if a == 612 else Fraction(0)))
    question2 = gvfs.Question(policy2, (lambda x,y,z: 1), (lambda x,y,z: 0 if x[0]>550 else 1))

    # meant to ask how many steps until the position is > 550 (on-policy)
    question3 = gvfs.Question(None, (lambda x, y, z: 0 if x[0] > 550 else 1),
                              (lambda x, y, z: 0 if x[0] > 550 else 1))

    # meant to ask the expected value of the position over the next ~10 timesteps
    question4 = gvfs.Question(None, lambda x,y,z: x[0], lambda x,y,z: 0.9)


    demons = []

    demon1 = gvfs.GeneralValueFunctionTDLambda(initial_on_features=on_features,
                                               question=question1,
                                               num_features=num_features,
                                               initial_weights=numpy.zeros(num_features),
                                               trace_decay=0.2)
    demons.append(demon1)

    demon2 = gvfs.GeneralValueFunctionGTD(initial_on_features=on_features,
                                          question=question2,
                                          num_features=num_features,
                                          initial_weights=numpy.zeros(num_features),
                                          trace_decay=0)
    demons.append(demon2)

    demon3 = gvfs.GeneralValueFunctionTDLambda(initial_on_features=on_features,
                                               question=question3,
                                               num_features=num_features,
                                               initial_weights=numpy.zeros(num_features),
                                               trace_decay=0.2)
    demons.append(demon3)

    demon4 = gvfs.GeneralValueFunctionTDLambda(initial_on_features=on_features,
                                               question=question4,
                                               num_features=num_features,
                                               initial_weights=numpy.zeros(num_features),
                                               trace_decay=0.2)

    demons.append(demon4)

    # Now we start the episode.

    for t in range(1, 1000+1):

        # take action a_t!
        action_old = behaviour_policy(sensor_readings, on_features)
        myRobot.take_action(action_old)

        # take new sensor readings (get s' = s_{t+1})
        sensor_readings = myRobot.get_readings()
        if len(sys.argv) > 1 and sys.argv[1] == 'tilecoding':
            on_features = simple_tiles.get_x(sensor_readings, corners[0], corners[1], num_bins)
        else:
            on_features = coding.get_x(sensor_readings, num_on_features, distance_metric='hamming')

        # update the state space
        for d in demons:
            if d.__class__ == gvfs.GeneralValueFunctionGTD:
                #TODO: Decide on what state being passed to get_probability should be!
                d.update(sensor_readings, on_features, t, action_old, behaviour_policy_keeper.get_probability(sensor_readings, action_old))
                # print('Action: ' + str(action_old) + ' Rho: ' + str(d.importance_ratio))
            else:
                d.update(sensor_readings, on_features, t)
            d.learn()

    # What is the answer at the following states?

    if len(sys.argv) > 1 and sys.argv[1] == 'tilecoding':
        print('35: ' + str(simple_tiles.get_x([0,0,35], corners[0], corners[1], num_bins)))
        feat_35 = simple_tiles.get_x([0,0,35], corners[0], corners[1], num_bins)
        feat_40 = simple_tiles.get_x([0,0,40], corners[0], corners[1], num_bins)
        print(sum(demon1.weights[feat_35]))
        print(sum(demon1.weights[feat_40]))
    else:
        # answers for question 2
        feat412 = coding.get_x([412, 0, 40], num_on_features=num_on_features, distance_metric='hamming')
        feat512 = coding.get_x([512, 0, 40], num_on_features=num_on_features, distance_metric='hamming')
        feat612 = coding.get_x([612, 0, 40], num_on_features=num_on_features, distance_metric='hamming')
        print(sum(demon2.weights[feat412]))
        print(sum(demon2.weights[feat512]))
        print(sum(demon2.weights[feat612]))
        print
        # answers for question 1
        print(sum(demon1.weights[coding.get_x([0, 0, 35], num_on_features=num_on_features, distance_metric='hamming')]))
        feat_35 = coding.get_x([0, 0, 35], num_on_features=num_on_features, distance_metric='hamming')
        print(sum(demon1.weights[coding.get_x([0, 0, 40], num_on_features=num_on_features, distance_metric='hamming')]))
        feat_40 = coding.get_x([0, 0, 40], num_on_features=num_on_features, distance_metric='hamming')
    print(set(feat_35).symmetric_difference(set(feat_40)))
    print(demon1.weights[list(set(feat_35).symmetric_difference(set(feat_40)))])


