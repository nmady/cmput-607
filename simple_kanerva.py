import numpy

import struct


def floatToBits(f):
    # http://stackoverflow.com/questions/14431170/get-the-bits-of-a-float-in-python
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]


class KanervaCoding():

    def __init__(self, low_corner, high_corner, num_prototypes, random_seed=None):
        numpy.random.seed(random_seed)

        assert len(low_corner) == len(high_corner)
        assert num_prototypes > 2

        self.num_dims = len(low_corner)
        self.num_prototypes = num_prototypes

        columns = [numpy.random.uniform(low_corner[dim], high_corner[dim], num_prototypes) for dim in range(self.num_dims)]
        columns = numpy.array(columns)
        self.prototypes = numpy.transpose(columns)

    def get_x(self, sensor_reading, num_on_features, distance_metric='euclidean'):

        assert len(sensor_reading) == self.num_dims

        if distance_metric is 'hamming':
            distances = []
            for proto in self.prototypes:
                # take the XOR and count the 1s, then sum over all 3 dimensions.
                distances.append(sum([bin(floatToBits(sensor_reading[dim]) ^ floatToBits(proto[dim])).count('1') for dim in range(self.num_dims)] ))
            distances = numpy.array(distances)
        else: #default to euclidean
            distances = numpy.array([numpy.linalg.norm(self.prototypes[i]-sensor_reading) for i in range(self.num_prototypes)])

        # Return the indices of the prototypes with the smallest distance from the sensor_reading
        return numpy.argpartition(-distances, -num_on_features)[-num_on_features:]



if __name__ == '__main__':
    low_corner = (0, 0, -10)
    high_corner = (1024, 2028, 90)
    sensor_reading = (110, 0, 39)
    #sensor_reading = (1023,2027,89)

    coding = KanervaCoding(low_corner, high_corner, 50, random_seed=10)

    print('Example:')
    print('low_corner = (0,0,-10)')
    print('high_corner = (1024,2028,90)')
    print('sensor_reading = ' + str(sensor_reading))

    #print(bin_readings(sensor_reading,low_corner,high_corner,10))
    #show_bins(low_corner,high_corner,10)

    print(coding.get_x(sensor_reading, 15, distance_metric='hamming'))