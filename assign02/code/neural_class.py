from __future__ import print_function
import neural_math
import neural_misc as misc


class NeuralNetwork():
    def __init__(self, learning_rate = 1, momentum_factor = 1, max_iterations = 100):
        #Class member declaration
        self.const_learning_rate = learning_rate
        self.const_momentum_factor = momentum_factor
        self.const_iterations = max_iterations
        self.network_length = 0
        self.nodes = []    
        self.weights = []
        self.errors = []
        self.weights_previous = []

    #Initialise nodes and errors list according to Neural Network structure 
    def init_network(self, neuron_data):
        #Empty lists
        self.nodes = []
        self.errors = []
        self.weights = self.weights_previous = []
        #Create nodes and errors list to store node output and error accordingly
        self.network_length = len(neuron_data)
        for layer in range(self.network_length):
            #List are created individually to point to different data
            #Store 1 initially
            self.nodes.append([1 for node in range(neuron_data[layer])])
            self.errors.append([1 for node in range(neuron_data[layer])])
        #Create weights list. Assume that all nodes are connected with all next layer nodes
        for  layer in range(self.network_length-1):
            nested_list = []
            #Initialize with small random value
            for nodes in range(neuron_data[layer]):
                nested_list.append([neural_math.randomNormalValue(0,0.2) for nodes_next_layer in range(neuron_data[layer+1])])
            self.weights.append(nested_list)
        #Append a final layer that will hold the bias weights
        self.weights.append([[neural_math.randomNormalValue(0,0.1) for nodes in range(neuron_data[layer+1])]for layer in range(self.network_length-1)])
        #Create a copy of weights list to store previous values 
        #Initially, previous weights are equal to current in order to cancel out the momentum
        self.weights_previous = [[self.weights[layer][input_node][:] for input_node in range(len(self.weights[layer]))] for layer in range(self.network_length)]

    #Print weights
    def show_weights(self):
        print("Weights")
        for element in self.weights:
            print(end = '\t')
            for subelement in element:
                print(end = '\t')
                misc.pprint(subelement)
        print("Previous weights")
        for element in self.weights_previous:
            print(end = '\t')
            for subelement in element:
                print(end = '\t')
                misc.pprint(subelement)

    #Prints network
    def show(self):
        print("Nodes:")
        for element in self.nodes:
            print(end = '\t')
            misc.pprint(element)
        print("Weights:")
        for element in self.weights:
            print(end = '\n')
            for subelement in element:
                print(end = '\t')
                misc.pprint(subelement)

    def get_output(self):
        return self.nodes[-1]
    def get_weights(self):
        return self.weights

                
    #The function assigns the inputs to the first layer nodes then runs a loop 
    #passing through every layer and calculates the output of each node
    def forward_pass(self, inputs, outputs):
        self.nodes[0] = inputs[:]
        #Forward pass by getting current and next layer index to do operations
        for current_layer in range(self.network_length-1):
            for output_node in range(len(self.nodes[current_layer+1])):
                #Node output is calculated by adding the weight*output of previous nodes
                #The bias is then added and sent to the activation(sigmoid) function
                #Add the bias value
                sum_of_inputs = self.weights[-1][current_layer][output_node]
                #Add the rest of the inputs       
                for input_node in range(len(self.nodes[current_layer])):
                    sum_of_inputs += self.weights[current_layer][input_node][output_node]*self.nodes[current_layer][input_node]
                self.nodes[current_layer+1][output_node] = neural_math.sigmoid(sum_of_inputs)
        #Return the error between final layer output and target
        return [output - target for output, target in zip(self.nodes[-1], outputs)]


    #Backward porpagation loop to calcualte error
    def backward_pass(self, target ):
        #Calculate output error in final layer, compared to the target for each output node
        for current_node in range(len(self.nodes[-1])):
            self.errors[-1][current_node] = neural_math.sigmoid_derivative(self.nodes[-1][current_node])*(target[current_node] - self.nodes[-1][current_node])
        #Loop backwards in the network and calculate ouputs of each hidden layer node in the hidden layers. If there are no hidden layers, it will not execute
        for current_layer in range(2, self.network_length+1):
            for current_node in range(len(self.nodes[-current_layer])):
                sum_of_outputs = sum(self.errors[1-current_layer][next_node]*self.weights[-current_layer][current_node][next_node] for next_node in range(len(self.nodes[1-current_layer])))
                self.errors[-current_layer][current_node] = neural_math.sigmoid_derivative(self.nodes[-current_layer][current_node])*sum_of_outputs
    

    def update_weights(self):
        #Calculate new weights - 3 dimension loop in list 'weights' 
        for current_layer in range(self.network_length-1):
            for current_node in range(len(self.nodes[current_layer])):
                for next_node in range(len(self.nodes[current_layer+1])):
                    #Find momentum
                    momentum = self.const_momentum_factor * (self.weights[current_layer][current_node][next_node] - self.weights_previous[current_layer][current_node][next_node])
                    #Calculate new weight, store in temporary variable
                    temp = self.weights[current_layer][current_node][next_node] + (self.const_learning_rate*self.errors[current_layer+1][next_node]*self.nodes[current_layer][current_node]) -  momentum
                    #Replace previous momentum values with current time
                    self.weights_previous[current_layer][current_node][next_node] = self.weights[current_layer][current_node][next_node]
                    #Set current time weight as new weight
                    self.weights[current_layer][current_node][next_node] = temp
        #Update bias weights (final weights layer)
        for current_layer in range(self.network_length-1):
            for current_node in range(len(self.nodes[current_layer+1])):
                momentum = self.const_momentum_factor * (self.weights[-1][current_layer][current_node] - self.weights_previous[-1][current_layer][current_node]) 
                temp = self.weights[-1][current_layer][current_node] + (self.const_learning_rate*self.errors[current_layer+1][current_node]) - momentum 
                self.weights_previous[-1][current_layer][current_node] = self.weights[-1][current_layer][current_node]
                self.weights[-1][current_layer][current_node] = temp


