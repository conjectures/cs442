import neural_misc as misc
import numpy as np

#Function reads parameters from filename. It is assumed that information is divided by newline
def read_rbf_params(filename):
    with open(filename,'r') as file:
        for line in file:
            if "maxIterations" in line:
                iterations = int(next(file).strip())
            if "dataFile" in line:
                datafile = next(file).strip()
            if "resultsFile" in line:
                resultsfile = next(file).strip()
            if "centersFile" in line:
                centersfile = next(file).strip()
            if "weightsFile" in line:
                weightsfile = next(file).strip()
    return [ iterations, datafile, resultsfile, centersfile, weightsfile]

def read_params(filename):
    with open(filename,'r') as file:
        for line in file:
            if "arrayWidth" in line:
                array_width = int(next(file).strip())
            if "arrayHeight" in line:
                array_height = int(next(file).strip())
            if "maxIterations" in line:
                iterations = int(next(file).strip())
    return [ array_width, array_height, iterations ]

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

#Functions used to read only numbers from data
#Below function reads a line and returns only the numbers within the list
def read_line_element(data):
    for element in data.split(','):
        number = misc.try_cast_float(element.strip(),None,ValueError)
        if number:
            yield number
        else:
            continue
#This function iterates over lines in file and returns only the numbers
def read_number_line(file_object):
    while True:
        data = file_object.readline()
        #Check if EOF
        if not data:
            break
        #Return a list of numbers
        yield [x for x in read_line_element(data)]
#This function reads a file and returns a list of only numbers
def read_numbers(filename):
    list_object = []
    file_object = open(filename, 'r')
    for line in read_number_line(file_object):
        list_object.append(line)
    file_object.close()
    return list_object

#Function that writes comma separated results in specified file
def write_result(filename, iterator, training_output, testing_output = None):
    with  open(filename, 'a') as result_file:
        if testing_output is None:
            result_file.write(",".join(str(item) for item in [iterator, training_output]))
        else:
            result_file.write(",".join(str(item) for item in [iterator, training_output, testing_output]))
        result_file.write("\n")

#Function used to write weights into file as comma separated values
def write_string(filename, data):
    with  open(filename, 'a') as result_file:
        result_file.write(data)

def write_array_data(filename, array):
    with open(filename, 'a') as result_file:
        for row in array:
            result_file.write(",".join(str(item) for item in row))
            result_file.write("\n")
def write_vector_data(filename, vector):
    with open(filename, 'a') as result_file:
        result_file.write(",".join(str(item) for item in vector))
        result_file.write("\n")
        

#Function returns normalise list of data from a file or list. Does not change initial list
def normalise_data(source):
    if isinstance(source, (str,bytes)):
        list_object = read_numbers(source)
    else:
        list_object = source
    #get transpose list to make columns into rows
    list_transpose = list(map(list, zip(*list_object)))
    for i in range(0, len(list_transpose)):
        #find min and max in each column
        maximum = float(max(list_transpose[i]))
        minimum = float(min(list_transpose[i]))
        for j in range(len(list_transpose[i])):
            list_transpose[i][j] = (list_transpose[i][j] - minimum)/(maximum - minimum)
    return list(map(list, zip(*list_transpose)))

#Normalise numpy array with built-in operations
def normalise_array(array):
    if type(array).__module__ == np.__name__ :
        array_min = np.amin(array, axis = 0)
        array_max = np.amax(array, axis = 0)
        return (array - array_min) /(array_max - array_min) 
    else:
       return  normalise_data(array)

