import matplotlib.pyplot as plt

#Plots training(y) and testing(z) results over epochs(x)
def plot(filename):
    data = []
    x = []
    y = []
    z = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(list(map(float, line.split(","))))
    for item in data: 
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])
    plt.ion()
    plt.figure(1)
    plt.plot(x, y, 'bs', x, z, 'g^')
    plt.show()

#Function used to plot the one of the weights over epochs
def plot2(filename, index):
    time = []
    weight = []
    data = []
    with open(filename,'r') as file:
        for line in file:
            line = line.split(",")
            data.append(list(map(float, line)))
    for item in range(len(data)):
        time.append(data[item][0])
        weight.append(data[item][index])
    plt.figure(2)
    plt.plot(time, weight) 
    plt.show()
    
