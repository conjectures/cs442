import math
import numpy as np
import file_io as io
import neural_math as nmath
import neural_draw as draw

def main():
#INITIALIZATION
    #Read parameters from file
    array_width, array_height, vector_length, max_epochs, train_file, test_file = io.read_params("parameters.txt")
    
    #Find number of neurons
    array_size = array_width*array_height
    #Set initial deviation
    standard_deviation = array_size*(0.5)
    #Set initial neighborhood
    n_init = nmath.randomValue()
    
    #Array operations

    #Functional arrays
    som = np.zeros(array_size, dtype=str).reshape(array_height,array_width)
    #Arrays containing the error
    train_error = np.zeros(max_epochs)
    test_error = np.zeros(max_epochs)
    #Make weight array:
    weights = np.random.rand(array_size, vector_length)
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
    #Import training and testing data
    training = []
    testing = []
    #Remove letter from input data 
    training_full = io.read_data(train_file)
    testing_full = io.read_data(test_file)
    for row in training_full:
        training.append(row[1:])
    for row in testing_full:
        testing.append(row[1:])
    #Copy lists into a numpy arrays
    test_data = np.asarray(testing)
    train_data = np.asarray(training)
   
   #Overwrite results files
    io.overwrite_file("results.txt")
    io.overwrite_file("clustering.txt")

#ITERATION
    for epoch in range(max_epochs):
        #Shuffle inputs 
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        #Find deviation and neighborhood size
        deviation = nmath.get_deviation(standard_deviation,epoch,max_epochs)
        n_size = nmath.get_neighbourhood(n_init,epoch,max_epochs)

        #TRAINING
        for pattern in range(np.size(train_data,0)):
            inputs = train_data[pattern,:]
            differences = weights - inputs
            temp_array = np.power(differences,2)
            nodes = temp_array.sum(axis = 1)
            #Find minimum distance and 1d index.
            distance_min = np.amin(nodes)
            min_index = np.argmin(nodes)
            #Convert 1d index to 2d (i and j)
            imin = math.floor(min_index/array_width)
            jmin = min_index % array_width
            #Accumulate normalised error per each epoch
            train_error[epoch]+= (distance_min**2)
            
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
            temp_vect = np.multiply(np.transpose(differences),neighbours)
            weights -= np.transpose(n_size*temp_vect)
        
        #TESTING
        for pattern in range(np.size(test_data,0)):
            inputs = test_data[pattern,:]
            differences = weights - inputs
            temp_array = np.power(differences,2)
            nodes = temp_array.sum(axis = 1)
            #Find minimum distance and 1d index.
            distance_min = np.amin(nodes)
            min_index = np.argmin(nodes)
            #Convert 1d index to 2d (i and j)
            imin = math.floor(min_index/array_width)
            jmin = min_index % array_width
            #Accumulate normalised error per each epoch
            test_error[epoch]+= (distance_min**2)
            #Update map
            som[int(imin),int(jmin)] = testing_full[pattern][0]
        
        #Print Results
        print("Epoch {}".format(epoch))
        print("Mean training error: {}".format(train_error[epoch]/np.size(train_data,0)))
        print("Mean testing error: {}".format(test_error[epoch]/np.size(test_data,0)))
        print(som)
        #Write results
        io.write_result("results.txt",epoch,train_error[epoch],test_error[epoch])
        
    #LABELING
    for node in range(np.size(nodes)):
        weight_vector = weights[node,:]
        temp = (test_data - weight_vector).sum(axis = 0)
        num = np.argmin(temp)
        som[int(imap[node]),int(jmap[node])] = testing_full[int(num)][0]
        #np.set_printoptions(threshold = np.inf)
    print(som)
    #WRITE MAP DATA 
    io.write_array_data("clustering.txt",som)

    #PLOTTING
    draw.plot_yz("results.txt","Errors per Epoch")
    while True:
        x = input()
        if x is 'q':
            break

if __name__ == "__main__":
    main()
