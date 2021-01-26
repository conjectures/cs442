import matplotlib.pyplot as plt
import file_io as io
import numpy as np

#Plots two variables (y,z) over x with data from list or file
def plot_yz(source, title = None):
    data = []
    if isinstance(source, (str,bytes)):
        data = io.read_data(source)
    else:
        data = source[:]

    data_transpose = list(map(list, zip(*data)))
    plt.ion()
    plt.figure()
    plt.plot(data_transpose[0], data_transpose[1], 'bs', data_transpose[0], data_transpose[2], 'g^')
    if isinstance(title, (str,bytes)):
        plt.title(title)
    plt.show()

def plot_numpy(data, title = None):
    if type(data).__module__ == np.__name__:
        plt.ion()
        plt.figure()
        plt.plot(data[0,:],data[1,:], 'bs', data[0,:],data[2,:], 'g^')
        if isinstance(title, (str,bytes)):
            plt.title(title)
        plt.show()
    else:
        plot_yz(data)

