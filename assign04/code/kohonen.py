import math
import numpy as np
import file_io as io
import neural_math as nmath
import neural_draw as draw
from random import shuffle

#To test map, pass weights list and lrate = 0
def kohonen_train(training, max_epochs, weights = None, lrate = None, array_size = None):
#INITIALIZATION
    np.set_printoptions(threshold = np.inf)
    #kohonen_results = 'kohonen_results.txt'
    kohonen_weights = 'kohonen_weights.txt'
    
    data_size = len(training)
    #If array size is given, assume the map is rectangular. Reassign value if not squarable
    if array_size:
        array_height = array_width = int(math.ceil(math.sqrt(array_size)))
        array_size = array_height * array_width
    else:
        #General Rule: Map size should be 5*sqrt(data_length)
        array_width = array_height = int(math.ceil(math.sqrt(20*math.sqrt(data_size))))
        array_size = array_width * array_height
 
    #Set initial deviation
    standard_deviation = array_size*(0.5)
    #If learning rate is not given, set from 0 to 0.5
    if lrate is None:
        lrate = (0.5)*nmath.randomValue()

    #print("Learning rate: {}".format(lrate))
        
    if type(training).__module__ == np.__name__:
        train_data = training
    else:
        train_data = np.asarray(training)
    vector_length = int(np.size(train_data,1))

    #Functional arrays
    #Arrays containing the error
    train_error = np.zeros(max_epochs)
    #Check if weight array is zero
    if weights is None:
        #Make weight array:
        weights = np.random.rand(array_size, vector_length)
        #print("Weights array shape: {}".format(np.shape(weights)))

    #Keep node distances in 1d array
    nodes = np.zeros(array_size)
    #imap and jmap are arrays that help broadcasting a 1d to 2d array
    imap = np.arange(array_size)/array_width
    imap = np.floor(imap)
    jmap = np.arange(array_size)%array_width

    #Below arrays are initialized in order to prevent dynamic allocation
    itemp = np.zeros(array_size)
    neighbours = np.zeros(array_size)
    differences = np.zeros((array_size, vector_length), dtype = 'float64')
    temp_array = np.zeros((array_size, vector_length), dtype = 'float64')
    temp_vect = np.zeros(array_size)
   #Overwrite results files
    #io.overwrite_file(kohonen_results)
    io.overwrite_file(kohonen_weights)

#ITERATION
    for epoch in range(max_epochs):
        #Shuffle inputs 
        np.random.shuffle(train_data)
        #Find deviation and neighborhood size
        deviation = nmath.get_deviation(standard_deviation,epoch,max_epochs)
        learn_rate = lrate*nmath.get_learn_rate(epoch,max_epochs)

        #TRAINING
        for pattern in range(np.size(train_data,0)):
            inputs = train_data[pattern,:]
            #Find the difference between the w matrix and the input x
            #Matrix w is the combination of all weight vectors for each node
            differences = weights - inputs
            # (w-x)^2 = (x-w)^2 
            temp_array = np.power(differences,2)
            #Calculate distance for each node
            nodes = temp_array.sum(axis = 1)
            #Find minimum distance and 1d index.
            min_dist = np.amin(nodes)
            min_index = np.argmin(nodes)
            #Convert 1d index to 2d (i and j)
            imin = math.floor(min_index/array_width)
            jmin = min_index % array_width
            #Accumulate normalised error per each epoch
            train_error[epoch]+= (min_dist**2)
            #Find distance in i axis 
            itemp = imap-imin
            itemp = np.power(itemp, 2)
            #Find distance in j axis
            jtemp = jmap-jmin
            jtemp = np.power(jtemp, 2)
            #Add distances and find delta_weight for each weight
            neighbours = itemp + jtemp
            neighbours = neighbours/((-2.0)*(deviation**2))
            neighbours = np.exp(neighbours)
            #UPDATE WEIGHTS
            temp_vect = np.multiply(differences, neighbours[:,None])
            weights -= (learn_rate*temp_vect)
        #Print Results
        #print("Epoch {}".format(epoch))
        #print("Mean training error: {}".format(train_error[epoch]/data_size))
        #Write Results
        #io.write_result(kohonen_results, epoch, train_error[epoch]/data_size)) 
    #Write final weights state 
    train_error = train_error/data_size
    io.write_array_data(kohonen_weights, weights)
    return (weights, np.mean(train_error))


#LABELING
#Input normalised data, labels and weights. Output is map of labels
def kohonen_label(data, labels, weights):
    io.overwrite_file('clustering.txt')
    #Convert to numpy type if not already
    if type(data).__module__ != np.__name__:
        data = np.asarray(data)
    ##Read weights
    #if isinstance(weights, (str,bytes)):
        ##Read data and normalise data if 'data' argument is filename
        #weights = io.read_data(kweights)
    if type(weights).__module__ != np.__name__:
        weights = np.asarray(weights)

    array_size = np.size(weights,0)
    array_height = array_width = int(math.sqrt(array_size))
    imap = np.arange(array_size)/array_width
    imap = np.floor(imap)
    jmap = np.arange(array_size)%array_width

    som = np.zeros(array_size, dtype= object).reshape(array_height,array_width)
    psom = np.zeros(array_size, dtype=int).reshape(array_height,array_width)

    #Iterate through nodes, find winning input, assign label
    for node in range(array_size):
        weight_vector = weights[node,:]
        temp = (data - weight_vector)
        temp = np.power(temp,2)
        temp = temp.sum(axis = 1)
        num = np.argmin(temp)
        #Store som as the labels
        som[int(imap[node]),int(jmap[node])] = labels[int(num)]
        #Store som as the label indexes in the data 
        psom[int(imap[node]),int(jmap[node])] = num
    #print(som)
    #Write data in file
    #io.write_array_data("clustering.txt",som)
    #Return label and position map
    return som, psom


#Get data vector and return the winning weight vector
def kohonen_winner(data_vector, weights):
    #Make into numpy if not already
    if type(data_vector).__module__ != np.__name__:
        data_vector = np.asarray(data_vector)
    if type(weights).__module__ != np.__name__:
        weights = np.asarray(weights)
    #Initialise list 
    differences = np.zeros_like(weights)
    temp_array = np.zeros_like(weights[:,0])
    #Find input-weight difference
    differences = weights - data_vector
    #Square differences
    differences = np.power(differences, 2)
    #Find difference sum
    temp_array = differences.sum(axis = 1)
    #Find winning weight
    mindex = np.argmin(temp_array)
    return mindex


