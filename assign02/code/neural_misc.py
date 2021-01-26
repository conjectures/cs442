from __future__ import print_function


class subclassfloat(float):
    def __repr__(self):
        return "{0:0.2f}".format(self)
    

#Return True or False if object is a list
def islist(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, (str,bytes))

def try_cast_float(argument, failure, *exceptions):
    try:
        return float(argument)
    except exceptions or ValueError:
        return failure() if callable(failure) else failure
    
def try_except(success, failure, *exceptions):
    try:
        return success() if callable(success) else success
    except exceptions or Exception:
        return failure() if callable(failure) else failure


#Deep copy a list and change type when possible
def deepcopy(obj, element_type = float):
    newlist = []
    for element in obj:
        if islist(element):
            newlist.append( deepcopy(element, element_type) )
        else:
            newlist.append(try_except(element_type(element), element, ValueError))
    return newlist

#Function prints list with custom format
def pprint(obj):
    if islist(obj):
        newlist = []
        newlist = deepcopy(obj, subclassfloat)
        print(newlist)
    else:
        print("{0:0.2f}".format(obj) if isinstance(obj, (int,float)) else obj) 



#function that flattens nested lists.
def flatten(list_object):
    for element in list_object:
        #check if element is another list
        if islist(element):
            #recursively check if next layer is a list
            for sub_element in flatten(element):
                yield sub_element
        else:
            #termination condition; element is not a list, return element
            yield element

#simulates digital electronics encoder
def encode(output_range, integer):
    if integer <= output_range:
        return list(1 if index == integer-1 else 0 for index in range(output_range))
    else:
        print("Error: Data encoding size error")
        return list([0 for element in range(output_range)])


def decode(list_obj):
    output = sum([index+1 for index in range(len(list_obj)) if list_obj[index]==1])
    return output



def winner_takes_all(list_obj):
    maximum = max(list_obj)
    if (list_obj).count(maximum) > 1:
        #Return empty list if more than 1 max
        return list([0 for element in list_obj])
    else:
        return list([int(element/maximum) for element in list_obj])


    
