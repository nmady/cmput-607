from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.axes


# http://stackoverflow.com/questions/34376656/matplotlib-create-real-time-animated-graph
class RegrMagic(object):
    """Mock for function Regr_magic()
    """
    def __init__(self, axes, list_of_MagicAxes, one_step_function):
        self.t = 0
        self.axes = axes
        self.list_of_MagicAxes = list_of_MagicAxes
        self.one_step_function = one_step_function

    def __call__(self):
        self.t += 1
        dictionary_of_new_data_points = self.one_step_function(self.t)

        returnable = []

        # First, let's append our x and y data for each thingy in the dictionary.
        for magic_axis in self.list_of_MagicAxes:
            for data_key in dictionary_of_new_data_points:
                magic_axis.data_on_axes[data_key]['x'].append(dictionary_of_new_data_points[data_key]['x'])
                magic_axis.data_on_axes[data_key]['y'].append(dictionary_of_new_data_points[data_key]['y'])

                # Now we have to make a list of new plotlines for each data
                # This is doing ax.plot(x, y, 'k'), more or less
                plotting_function = magic_axis.data_on_axes[data_key]['plotting_func']
                returnable.append( plotting_function(self.axes[magic_axis.title],
                                                    magic_axis.data_on_axes[data_key]['x'],
                                                    magic_axis.data_on_axes[data_key]['y'],
                                                    magic_axis.data_on_axes[data_key]['kwargs']))

        return returnable


class MagicAxes(object):
    def __init__(self, title, position=111, ylabel=None, xlabel=None):
        self.title = title
        self.position = position
        self.ylabel = ylabel
        self.xlabel = xlabel

        self.data_on_axes = {}

    def add_data(self, name, x=[], y=[], plotting_func=matplotlib.axes.Axes.plot, kwargs='k'):
        self.data_on_axes[name] = {'kwargs': kwargs, 'plotting_func': plotting_func, 'x':x, 'y':y}


class MagicPlot(object):

    def __init__(self, list_of_MagicAxes, one_step_function):

        self.figure = plt.figure()
        self.axes = {}
        self.one_step_function = one_step_function

        for axis_data in list_of_MagicAxes:
            ax = self.figure.add_subplot(axis_data.position)
            ax.set_title(axis_data.title)
            #if axis_data.ylabel is not None:
            ax.set_ylabel(axis_data.ylabel)
            ax.set_xlabel(axis_data.xlabel)
            self.axes[axis_data.title] = ax

        self.regr_magic = RegrMagic(self.axes, list_of_MagicAxes, one_step_function)

        # I've got no idea what interval does.
        anim = animation.FuncAnimation(self.figure, animate, frames=self.frames, interval=1000)
        plt.tight_layout()
        plt.show()

    def frames(self):
        while True:
            yield self.regr_magic()


def animate(args):
    return tuple(args)

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
           ax_temperature.plot(t, s0_temperature, color='orange'), ax_temperature.plot(t, s1_temperature,
                                                                                       color='k'), \
           ax_voltage.plot(t, s0_voltage, color='orange'), ax_voltage.plot(t, s1_voltage, color='k'),