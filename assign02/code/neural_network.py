import neural_misc as misc
import file_io as io
import neural_class
import neural_draw
import neural_math          
        
def main():
##INITIALIZATION## 
    #Declare variables
    neuron_data, learning_rate, momentum_factor, iterations, file_train, file_test = io.read_params('parameters.txt')

    #Initialize network(class)
    network = neural_class.NeuralNetwork(learning_rate, momentum_factor, iterations)
    network.init_network(neuron_data)
    network.show()

    #Clear files or create if they don't exist
    io.overwrite_file("success.txt")
    io.overwrite_file("errors.txt")
    io.overwrite_file("weights.txt")
    #Read training/testing data
    training_data = io.read_data(file_train)
    testing_data = io.read_data(file_test)
    #Do a test and training loop in each epoch
    for epoch in range(iterations):
        #Initialize outputs
        print("\tEpoch: {}".format(epoch))
        testing_output = []
        training_output = []
        #Shuffle inputs
        neural_math.random.shuffle(training_data)
        neural_math.random.shuffle(testing_data)
        #Initialize success counters
        train_success, test_success = 0, 0
        train_success_ratio, test_success_ratio = 0, 0
        #Functional lists
        testing_error_flat = []
        training_error_flat = []
        
        #TRAINING
        for training_set in range(len(training_data)):
            target = []
            #Load target output from data
            target = misc.encode(neuron_data[-1],ord(training_data[training_set][0])-64)
            training_output.append(network.forward_pass(training_data[training_set][1:], target ))
            #Record success 
            if misc.winner_takes_all(network.get_output()) == target:
                train_success += 1
        #NETWORK FITTING
           #Calculate error due to each node with a back propagation
            network.backward_pass(target)
            #Update weights according to node errors and outputs
            network.update_weights()

        #TESTING
        for testing_set in range(len(testing_data)):
            target = []
            #Load target output from data
            target = misc.encode(neuron_data[-1],ord(testing_data[testing_set][0])-64)
            #Calculate error individually for each output node and store in list
            testing_output.append(network.forward_pass(testing_data[testing_set][1:], target))
            if misc.winner_takes_all(network.get_output()) == target:
                test_success += 1
    

    #CALCULATE RESULTS
        test_success_ratio = test_success/float(len(testing_data))
        train_success_ratio =  train_success/float(training_set+1)
        #Flatten nested list for easier manipulation
        testing_error_flat = list(misc.flatten(testing_output))
        training_error_flat = list(misc.flatten(training_output))
        testing_average_error = sum(map(abs, testing_error_flat))/float(len(testing_error_flat))
        training_average_error = sum(map(abs, training_error_flat))/float(len(training_error_flat))
        io.write_result("errors.txt", epoch, training_average_error, testing_average_error)
        io.write_result("success.txt", epoch, train_success_ratio, test_success_ratio)
    #Record final weights
    io.write_weights("weights.txt", network.get_weights())
    

##VISUALISATION##
    #Plot of results(Error vs epoch)
    neural_draw.plot_yz("success.txt", "SuccessRate per Epoch")
    neural_draw.plot_yz("errors.txt", "Errors per Epoch")
    
    while True:
        x = input()
        if x is 'q':
            break

if __name__ == "__main__":
    main()
