from  __future__ import print_function 
import numpy as np
import file_io as io
import neural_misc as misc
import neural_math as nmath
import neural_draw as draw
import kohonen as som
import math


def train_kohonen(inputs):
#Train Kohonen map
    error = []
    lrate = 0.6
    #First kohonen to initialise weights
    weights = som.kohonen_train(inputs, 100)[0]
    #Kohonen loop to minimise error
    index = 0
    while True:
        weights, temp = som.kohonen_train(inputs, 50 ,weights, lrate )

        #som.kohonen_label(data, weights)
        print("Iteration: {}, error: {}".format(index,temp))
        error.append(temp)
        if len(error) > 2:
            if lrate < 0.05:
                break
            if error[index] > error[index-1]:
                lrate = lrate*0.8
        index+=1
    return weights

def main():
#Read and process data
    max_iterations, data_file, resutls_file, centers_file = io.read_rbf_params('parameters.txt') 
    print(data_file)
    io.overwrite_file(resutls_file)
    data_full = io.read_data(data_file)
    #Make into numpy
    data_full = np.asarray(data_full)
    dsize = np.size(data_full,axis = 0)
    #Shuffle data by rows
    np.random.shuffle(data_full)
    #Train-Test Ratio
    ttratio = 21/31.0
    #Set train and testing lists
    training_full = data_full[:int(math.floor(ttratio*dsize)),:]
    testing_full = data_full[int(math.ceil(ttratio*dsize)):,:]
    #Set training data
    labels = training_full[:,0]
    targets = (training_full[:,1]).astype(float)
    inputs = (training_full[:,2:]).astype(float)
    inputs = io.normalise_array(inputs)
    #Set testing data
    ttargets = (testing_full[:,1]).astype(float)
    tinputs = (testing_full[:,2:]).astype(float)
    tinputs = io.normalise_array(tinputs)

    data_size = np.size(inputs, axis = 0)
    tdata_size = np.size(tinputs, axis = 0)
    #print("Data size: {}".format(data_size))

    #Train kohonen with training inputs if weight were not provided
    centers = io.read_data(centers_file)
    if not centers:
        print("Kohonen map training:")
        weights = train_kohonen(inputs)
        #Label kohonen
        kmap, pmap = som.kohonen_label(inputs,labels, weights)
        print("Kohonen map:")
        print(kmap)
        print("Label position map:")
        print(pmap)
        #Count kohnonen nodes
        unique_labels = np.unique(kmap)
        unique_positions = np.unique(pmap)

        #print(unique_labels)
        print(unique_positions)
        #print("number of unique elements: {}".format(np.size(unique_labels)))

        #Find closest weight for each unique label
        closest = []
        for plabel in unique_positions:
            closest.append(som.kohonen_winner(inputs[plabel,:], weights))
            #print("Closest weight index: {}".format(closest[-1])) 
        closest = np.asarray(closest)
        #Some weights are repeated
        closest = np.unique(closest)
        #Extract weights and use as rbf centers
        centers = np.take(weights, closest, axis = 0)
        #Overwrite file and write new centers in file:
        io.overwrite_file(centers_file)
        io.write_array_data(centers_file, centers)
    #Find number of centers
    if type(centers).__module__ != np.__name__:
        centers = np.asarray(centers)
    csize = np.size(centers, axis= 0)
    print("Number of centers: {}".format(csize)) 

    #RBF
#Initialise sigmas, coeffs, learning rates
    coeffs = np.ones(csize) 
    sigmas = np.ones(csize)
    #Learning rates no larger than 0.5
    clrate = (0.01)*nmath.randomValue()
    rlrate = (0.02)*nmath.randomValue()
    slrate = (0.02)*nmath.randomValue()
    #Calculate inverse sigmas
    invsigmas = 1./sigmas
    #print(sigmas)
    #print(invsigmas)


    #supplementary arrays
    temp = np.zeros(csize)
    dist = np.zeros(csize)
    differences = np.zeros_like(centers)
    outputs = np.zeros(data_size)
    errors = np.zeros(data_size)
    toutputs = np.zeros(tdata_size)
    terrors = np.zeros(tdata_size)
    train_errors = np.zeros(max_iterations)
    test_errors = np.zeros(max_iterations)
        
    #Debugging
    coeffs_sum = np.zeros(max_iterations)
    centers_sum = np.zeros(max_iterations)
    sigmas_sum = np.zeros(max_iterations)

    for epoch in range(max_iterations):
    #Testing
        for pattern in range(tdata_size):
        #Find Outputs:
            differences = centers - tinputs[pattern,:]
            distance = (differences**2).sum(axis = 1)
            basis_function = np.exp(-distance/(2.*(sigmas**2)))
            toutputs[pattern] = (basis_function* coeffs).sum()
            #print("Output: {}\tExpected: {}".format(toutputs[pattern],ttargets[pattern]))
            terrors[pattern] = ttargets[pattern] - toutputs[pattern]
            #print("Error: {}".format(terrors[pattern]))

    #Training
        for pattern in range(data_size):
        #Find Outputs:
            differences = centers - inputs[pattern,:]
            distance = (differences**2).sum(axis = 1)
            basis_function = np.exp(-distance/(2.*(sigmas**2)))
            outputs[pattern] = (basis_function* coeffs).sum()
            #print("Output: {}\tExpected: {}".format(outputs[pattern],targets[pattern]))
            errors[pattern] = targets[pattern] - outputs[pattern]
            
        #Update variables:
            #coeffs
            coeffs = coeffs - basis_function*(clrate*errors[pattern])
            #centers
            temp = basis_function*coeffs*(invsigmas**2)*errors[pattern]*rlrate
            #print(temp)
            centers = centers - differences*temp[np.newaxis,:].T
            #sigmas
            sigmas = sigmas + basis_function*distance*(coeffs*(invsigmas**3)*slrate*errors[pattern])
            invsigmas = 1./sigmas


        train_errors[epoch] = np.mean(errors**2)
        test_errors[epoch] = np.mean(terrors**2)
        print("Epoch: {}\tTrain error: {}\tTest error: {}".format(epoch, train_errors[epoch],test_errors[epoch]))
        #Write results
        io.write_result(resutls_file, epoch, train_errors[epoch],test_errors[epoch])

    #Reduce learning rate algorithm:
        #Apply more than 5 epochs have passed
        #if epoch > 100:
        #    if train_errors[epoch] >= train_errors[epoch-1]:
        #        clrate = clrate*0.5
        #        rlrate = rlrate*0.5
        #        slrate = slrate*0.5
            #Check if overfitting condition
            #if train_errors[epoch] < train_errors[epoch-1] and test_errors[epoch] > test_errors[epoch-1]:
                #break
        
     
        #coeffs_sum[epoch] = coeffs.sum()
        #centers_sum[epoch] = (centers.sum(axis = 1)).sum()
        #sigmas_sum[epoch] = sigmas.sum()
    #draw.plot_numpy(np.vstack((np.arange(max_iterations), coeffs_sum , coeffs_sum)),"Coeffs")
    #draw.plot_numpy(np.vstack((np.arange(max_iterations), centers_sum, centers_sum)),"Centers")
    #draw.plot_numpy(np.vstack((np.arange(max_iterations), sigmas_sum, sigmas_sum)),"Sigmas")
    draw.plot_numpy(np.vstack((np.arange(max_iterations),train_errors, test_errors)))
    raw_input()
    return 
    

if __name__ == "__main__":
    main()
