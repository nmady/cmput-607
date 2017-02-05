import fake_robot as robot
import general_value_functions as gvfs
import simple_tiles
import simple_kanerva
import sys

import numpy

if __name__ == '__main__':
    myRobot = robot.Robot()

    t = 0

    sensor_readings = myRobot.get_readings()
    corners = myRobot.get_sensorspace_corners()

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

    print('init',sensor_readings,on_features)

    #meant to ask how many steps uptil the temp is 40
    question1 = gvfs.Question(None, (lambda x,y,z: 0 if x[2]==40 else 1), (lambda x,y,z: 0 if x[2]==40 else 1))

    demon1 = gvfs.GeneralValueFunctionTDLambda(initial_on_features=on_features,
                                               question=question1,
                                               num_features=num_features,
                                               initial_weights=numpy.zeros(num_features),
                                               trace_decay=0.2)

    for t in range(1, 10000+1):

        # take new sensor readings (get s' = s_{t+1})
        sensor_readings = myRobot.get_readings()
        if len(sys.argv) > 1 and sys.argv[1] == 'tilecoding':
            on_features = simple_tiles.get_x(sensor_readings, corners[0], corners[1], num_bins)
        else:
            on_features = coding.get_x(sensor_readings, num_on_features, distance_metric='hamming')

        # update the state space
        demon1.update(sensor_readings, on_features, t)
        demon1.learn()


    # What is the answer at the following states?
    if len(sys.argv) > 1 and sys.argv[1] == 'tilecoding':
        print('35: ' + str(simple_tiles.get_x([0,0,35], corners[0], corners[1], num_bins)))
        feat_35 = simple_tiles.get_x([0,0,35], corners[0], corners[1], num_bins)
        feat_40 = simple_tiles.get_x([0,0,40], corners[0], corners[1], num_bins)
        print(sum(demon1.weights[feat_35]))
        print(sum(demon1.weights[feat_40]))
    else:
        print(sum(demon1.weights[coding.get_x([0, 0, 35], num_on_features=num_on_features, distance_metric='hamming')]))
        feat_35 = coding.get_x([0, 0, 35], num_on_features=num_on_features, distance_metric='hamming')
        print(sum(demon1.weights[coding.get_x([0, 0, 40], num_on_features=num_on_features, distance_metric='hamming')]))
        feat_40 = coding.get_x([0, 0, 40], num_on_features=num_on_features, distance_metric='hamming')
    print(set(feat_35).symmetric_difference(set(feat_40)))
    print(demon1.weights[list(set(feat_35).symmetric_difference(set(feat_40)))])