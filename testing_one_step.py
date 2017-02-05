from magic_plotting import MagicPlot, MagicAxes

def my_one_step(time):
    return { 'A line': {'x': time, 'y': time} }

if __name__ == '__main__':

    myMA = MagicAxes('My Title')
    myMA.add_data('A line')

    myMP = MagicPlot([myMA], my_one_step)