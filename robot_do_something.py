

# 1) Plug in power to the robot's barrell connector
# 2) Plug the USB2Dynamixel (set to TTL) into the USB port of your computer
# 3a) Plug the cable into the proximal servo; make sure only *one* servo is connected to the USB2Dynamixel
# 3b) Plug in servo cable to USB2Dynamixel
# 4) Initialize and try out the first servo as follows

from lib_robotis_hack import *
import sys, signal
import numpy
import random
import time
import csv

from matplotlib import pyplot as plt
from matplotlib import animation

#http://stackoverflow.com/questions/18994912/ending-an-infinite-while-loop

def signal_handler(signal, frame):
    print("\n Program exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

then = str(time.clock())
with open(then+'.csv','w') as csvfile:
    csvfile.write('#time,s0_command,s1_command,s0_angle,s1_angle,s0_load,s1_load,s0_temp,s1_temp,s0_voltage,s1_voltage,\n')


# Create the USB to Serial channel
# Use the device you identified above, and baud of 1Mbps
D = USB2Dynamixel_Device(dev_name=sys.argv[1],baudrate=1000000)

# Identify servos on the bus (should be only ONE at this point)
# Should return: "FOUND A SERVO @ ID 1"
s_list = find_servos(D)

s0 = Robotis_Servo(D,s_list[0])
s1 = Robotis_Servo(D,s_list[1])


class RegrMagic(object):
    """Mock for function Regr_magic()
    """
    def __init__(self):
        self.t = 0
    def __call__(self):
        s0.move_angle(1.5*numpy.sin(0.1*self.t))
        s1.move_angle(numpy.sin(self.t))
        self.t += 1

        args = [s0.read_angle(), s1.read_angle(), \
            s0.read_load(), s1.read_load(), \
            s0.read_temperature(), s1.read_temperature(), \
            s0.read_voltage(), s1.read_voltage()]

        with open(then + '.csv', 'a') as csvfile:
            csvfile.write(str(self.t)+',move_angle('+str(1.5*numpy.sin(0.1*self.t))+'),move_angle('+str(numpy.sin(self.t))+'),'+str(args[0])+','+str(args[1])+','+str(args[2])+','+str(args[3])+','+str(args[4])+','+str(args[5])+','+str(args[6])+','+str(args[7])+',\n')

        return self.t, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
            

regr_magic = RegrMagic()

def frames():
    while True:
        yield regr_magic()

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

t = []
s0_position = []
s1_position = []
s0_load = []
s1_load = []
s0_temperature = []
s1_temperature = []
s0_voltage = []
s1_voltage = []
def animate(args):
    t.append(args[0])
    s0_position.append(args[1])
    s1_position.append(args[2])
    s0_load.append(args[3])
    s1_load.append(args[4])
    s0_temperature.append(args[5])
    s1_temperature.append(args[6])
    s0_voltage.append(args[7])
    s1_voltage.append(args[8])
    return ax_position.plot(t, s0_position, color='orange'), ax_position.plot(t, s1_position, color='k'), \
        ax_load.plot(t, s0_load, color='orange'), ax_load.plot(t, s1_load, color='k'), \
        ax_temperature.plot(t, s0_temperature, color='orange'), ax_temperature.plot(t, s1_temperature, color='k'), \
        ax_voltage.plot(t, s0_voltage, color='orange'), ax_voltage.plot(t, s1_voltage, color='k'), 

# I've got no idea what interval does.
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000)
plt.tight_layout()
plt.show()


print('Done')

