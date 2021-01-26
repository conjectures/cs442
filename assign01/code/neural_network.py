import file_io
import neural_class
import neural_draw
        
        
def main():
##INITIALIZATION## 
    #Declare variables
    neuron_data, learning_rate, momentum_factor, iterations, file_train, file_test, file_results = file_io.readParams('parameters.txt')

    #Initialize network(class)
    network = neural_class.NeuralNetwork(learning_rate, momentum_factor, iterations)
    network.init_network(neuron_data)
    network.show()

    #Clear files or create if they don't exist
    file_io.overwriteFile(file_results)
    file_io.overwriteFile("weights.txt")
    #Read training/testing data
    training_data = file_io.readData(file_train)
    testing_data = file_io.readData(file_test)
    #Do a test and training loop in each epoch
    for epoch in range(iterations):
        testing_output = []
        training_output = []
        weights = []
        #Run testing data through neural network and write data in list. Find testing output first, before the network fitting.
        for testing_set in range(len(testing_data)):
            testing_output.append(network.forward_pass(testing_data[testing_set]))
        for training_set in range(len(training_data)):
            #Repeat for training data
            training_output.append(network.forward_pass(training_data[training_set])) 
            #Network fitting
           #Node outputs are stored in the nodes list that was calculated on forward pass
            network.backward_pass(training_data[training_set])
            #Update weights according to node errors
            network.update_weights()
        #Calculate testing and training average error
        testing_average_error = (sum(sum(map(abs,x)) for x in testing_output))/float(len(testing_output))
        training_average_error = (sum(sum(map(abs,x)) for x in training_output))/float(len(training_output))
        file_io.writeResult(file_results, epoch, abs(training_average_error), abs(testing_average_error))

    #Transfer weights from nested lists to a single list(for plotting )
        for i in range(len(network.weights)):
            for j in range(len(network.weights[i])):
                for k in range(len(network.weights[i][j])):
                    weights.append(network.weights[i][j][k])
        #Records weights every epoch. Uncomment for plot
        #file_io.writeWeights("weights.txt", epoch, weights)
    #Record final weights
    file_io.writeWeights("weights.txt", epoch, weights)

##VISUALISATION##
    #Plot of results(Error vs epoch)
    neural_draw.plot(file_results)
    #User interaction loop
    while True:
        print("Please put inputs in the console. Press q to terminate")
        try:
            x = raw_input()
            input_a = int(x)
        #Input validation
        except ValueError:
            if (len(x)) == 0:
                continue
            #Exit condition
            if ord(x) == 113:
                return
            continue
        try:
            x = raw_input()
            input_b = int(x)
        #Input validation
        except ValueError:
            if (len(x)) == 0:
                continue
            #Exit condition
            if ord(x) == 113:
                return
            continue

        print("Output is:")

        output = network.forward_pass([input_a, input_b,0])
        if (output[0] >= 0.5):
            print(1)
        else: print(0)
        print("Exact value: %2.5f\n\n"%output[0])


if __name__ == "__main__":
    main()
