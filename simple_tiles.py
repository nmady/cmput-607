import numpy

def get_x(sensor_reading, low_corner, high_corner, num_bins):

    #sensor_reading, low_corner, and high_corner all have the same number of entries
    #num_bins either has the same number of entries as sensor_reading etc. OR is integer (use same for all dims)

    assert len(sensor_reading) == len(low_corner) == len(high_corner)
    assert type(num_bins) == int #or len(num_bins) == len(sensor_reading)

    binned = bin_readings(sensor_reading, low_corner, high_corner, num_bins)
    num_dims = len(binned)

    # So we're going to have a total of num_dims + 1 "on" features
    x = []

    # The first "on" feature is in num_dims dimensional space, where there are num_bins^num_dims features
    # For example, bins of 10 means 10^3-1 zeros and 1 one
    x_1 = 0
    for dim in range(1,num_dims):
        x_1 += (binned[-dim] - 1) * num_bins**(num_dims-dim)
    x_1 += binned[0]

    x.append(x_1)

    # The rest of the "on" features are just giving the bin for each dimension, regardless of other dimensions
    for dim in range(num_dims):
        x.append(num_bins**num_dims + dim*num_bins + binned[dim])

    # right now, x is 1-indexed, but we want 0-indexed 'cos python.

    return numpy.array(x)-1


def get_num_features(num_bins, num_dims):
    return num_bins**num_dims + num_bins*num_dims


def bin_readings(sensor_reading, low_corner, high_corner, num_bins):

    #sensor_reading, low_corner, and high_corner all have the same number of entries
    #num_bins either has the same number of entries as sensor_reading etc. OR is integer (use same for all dims)

    assert len(sensor_reading) == len(low_corner) == len(high_corner)
    assert type(num_bins) == int #or len(num_bins) == len(sensor_reading)

    num_dims = len(sensor_reading)

    binned = []

    for dim in range(num_dims):
        this_bin = numpy.digitize(sensor_reading[dim], numpy.array([low_corner[dim]
                                 + (high_corner[dim]-low_corner[dim])*j//num_bins
                                 for j in range(num_bins+1)]), right=False)

        binned.append(int(this_bin))

    return binned


def show_bins(low_corner, high_corner, num_bins):

    num_dims = len(low_corner)

    print([[low_corner[dim]
     + (high_corner[dim] - low_corner[dim]) * j // num_bins
     for j in range(num_bins + 1)] for dim in range(num_dims)])


if __name__ == '__main__':
    low_corner = (0, 0, -10)
    high_corner = (1024, 2028, 90)
    #sensor_reading = (110, 0, 38)
    #sensor_reading = (1023,2027,89)
    sensor_reading = (0,0, 40)

    print('Example:')
    print('low_corner = (0,0,-10)')
    print('high_corner = (1024,2028,90)')
    print('sensor_reading = ' + str(sensor_reading))
    print('num_bins=10')

    print(bin_readings(sensor_reading,low_corner,high_corner,10))
    show_bins(low_corner,high_corner,10)

    print(get_x(sensor_reading, low_corner, high_corner, 10))