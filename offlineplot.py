# AUTHOR: Nadia M. Ady


import sys, signal
import numpy
import time

from matplotlib import pyplot as plt

#http://stackoverflow.com/questions/18994912/ending-an-infinite-while-loop

def signal_handler(signal, frame):
    print("\n Program exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

fig = plt.figure()
ax_position = fig.add_subplot(411)
ax_position.set_title("Position (angle) vs. Timestep")
ax_position.set_ylabel("Angle (radians)")
ax_load = fig.add_subplot(412)
ax_load.set_title("Load vs. Timestep")
ax_load.set_ylabel("Load ($\propto$ Nm)")
ax_temperature = fig.add_subplot(413)
ax_temperature.set_title("Temperature vs. Timestep")
ax_temperature.set_ylabel("Temperature ($^\circ$C)")
ax_voltage = fig.add_subplot(414)
thingy = ax_voltage.set_title("Voltage vs. Timestep")
ax_voltage.set_ylabel("Voltage (V)")
ax_voltage.set_xlabel("Time (steps)")


data = numpy.loadtxt(sys.argv[1], delimiter=',', usecols=[0,3,4,5,6,7,8,9,10])
data = numpy.transpose(data)
print(data)

ax_position.plot(data[0], data[1], color='orange')
ax_position.plot(data[0], data[2], color='k')
ax_load.plot(data[0], data[3], color='orange')
ax_load.plot(data[0], data[4], color='k')
ax_temperature.plot(data[0], data[5], color='orange')
ax_temperature.plot(data[0], data[6], color='k')
ax_voltage.plot(data[0], data[7], color='orange')
ax_voltage.plot(data[0], data[8], color='k')

plt.tight_layout()

plt.show()
sys.exit(0)  


