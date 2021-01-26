#Function reads parameters from filename. It is assumed that information is divided by newline
def readParams(filename):
    with open(filename,'r') as file:
        for line in file:
            if "numNeurons" in line:
                neuron_data = list(map(int, (file.next().strip()).split(",")))
            if "learningRate" in line:
                learning_rate = float(file.next().strip())    
                #print("learning rate: %2.5f"%const_learning_rate)
            if "momentum" in line:
                momentum = float(file.next().strip())
            if "maxIterations" in line:
                iterations = int(file.next().strip())
            if "trainFile" in line:
                file_train = file.next().strip()
            if "testFile" in line:
                file_test = file.next().strip()
            if "resultFile" in line:
                file_result = file.next().strip()
    return [ neuron_data, learning_rate,momentum, iterations, file_train, file_test, file_result ]

#Used to clear (result)file before experiment
def overwriteFile(filename):
    open(filename,'w').close()

#Function used to read file line by line. Returns list generator
def readDataLine(file_object):
    while True:
        data = file_object.readline()
        #Check if EOF
        if not data:
            break
        yield list(map(int, data.split(",")))

#Separate data read function used to open file once and call readDataLine function
def readData(filename):
    list_object = []
    file_object = open(filename, 'r')
    for line in readDataLine(file_object):
        list_object.append(line)
    return list_object

#Function that writes comma separated results in specified file
def writeResult(filename, iterator, training_output, testing_output):
    with  open(filename, 'a') as result_file:
        result_file.write(",".join(str(item) for item in [iterator, training_output, testing_output]))
        result_file.write("\n")

#Function used to write weights into file as comma separated values
def writeWeights(filename, iterator, data):
    with  open(filename, 'a') as result_file:
        result_file.write(str(iterator))
        result_file.write(",")
        result_file.write(",".join(str(item) for item in data))
        result_file.write("\n")
    
