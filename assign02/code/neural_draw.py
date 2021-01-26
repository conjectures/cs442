import matplotlib.pyplot as plt
import file_io as io

#Plots two variables (y,z) over x with data from list or file
def plot_yz(source, title = None):
    data = []
    if isinstance(source, (str,bytes)):
        print("Source is filename")
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

##Function used to plot the one of the weights over epochs
#def plot_error(filename, index):
#    time = []
#    weight = []
#    data = []
#    with open(filename,'r') as file:
#        for line in file:
#            line = line.split(",")
#            data.append(list(map(float, line)))
#    for item in range(len(data)):
#        time.append(data[item][0])
#        weight.append(data[item][index])
#    plt.figure(2)
#    plt.plot(time, weight) 
#    plt.show()
#    
