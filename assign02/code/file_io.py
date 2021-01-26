import neural_misc as misc

#Function reads parameters from filename. It is assumed that information is divided by newline
def read_params(filename):
    with open(filename,'r') as file:
        for line in file:
            if "numNeurons" in line:
                neuron_data = list(map(int, (next(file).strip()).split(",")))
            if "learningRate" in line:
                learning_rate = float(next(file).strip())    
            if "momentum" in line:
                momentum = float(next(file).strip())
            if "maxIterations" in line:
                iterations = int(next(file).strip())
            if "trainFile" in line:
                file_train = next(file).strip()
            if "testFile" in line:
                file_test = next(file).strip()
            
    return [ neuron_data, learning_rate,momentum, iterations, file_train, file_test ]

#Used to clear (result)file before experiment
def overwrite_file(filename):
    open(filename,'w').close()

#Function used to read file line by line. Returns list generator
def read_data_line(file_object):
    while True:
        data = file_object.readline()
        #Check if EOF
        if not data:
            break
        #Try returning a list of floats or strings if casting doesn't work 
        yield [(misc.try_cast_float(x.strip(), x.strip(), ValueError)) for x in data.split(',') ]

#Separate data read function used to open file once and call read_data_line function
def read_data(filename):
    list_object = []
    file_object = open(filename, 'r')
    for line in read_data_line(file_object):
        list_object.append(line)
    file_object.close()
    return list_object

#Function that writes comma separated results in specified file
def write_result(filename, iterator, training_output, testing_output):
    with  open(filename, 'a') as result_file:
        result_file.write(",".join(str(item) for item in [iterator, training_output, testing_output]))
        result_file.write("\n")

#Function used to write weights into file as comma separated values
def write_weights(filename, data):
    with  open(filename, 'a') as result_file:
        result_file.write("\n".join(str(item) for item in data))

def write_array_data(filename, array):
    with open(filename, 'w') as result_file:
        for row in array:
            result_file.write(",".join(str(item) for item in row))
            result_file.write("\n")

#Function returns normalise list of data from a file or list. Does not change initial list
def normalise_data(source):
    if isinstance(source, (str,bytes)):
        list_object = read_data(source)
    else:
        list_object = source
    #get transpose list to make columns into rows
    list_transpose = list(map(list, zip(*list_object)))
    for i in range(1, len(list_transpose)):
        #find min and max in each column
        maximum = float(max(list_transpose[i]))
        minimum = float(min(list_transpose[i]))
        for j in range(len(list_transpose[i])):
            list_transpose[i][j] = (list_transpose[i][j] - minimum)/(maximum - minimum)
    return list(map(list, zip(*list_transpose)))
    
