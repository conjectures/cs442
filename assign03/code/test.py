from  __future__ import print_function 
import numpy as np
import file_io as io


array_width, array_height, vector_length, learning_rate, iterations, train_file, test_file = io.read_params("parameters.txt")

print(array_width, array_height, vector_length, learning_rate, iterations, train_file, test_file)

a = np.array([2,3,4])
print(a.dtype, end='\n\n')

a = np.array([[1,2],[3,4]], dtype = complex)
print(a, end='\n\n')

#array of ones
a = np.ones((3,4), dtype ='int64')
print(a,end='\n\n')

#arange
a = np.arange(0,2,0.3)
print(a,end='\n\n')
#linspace
a = np.linspace(0,2,9)
print(a,end='\n\n')
#Printing options
a = np.arange(10000).reshape(100,100)
print(a,end='\n\n')
#np.set_printoptions(threshold='nan')
#print(a,end='\n\n')

#Basic Operations
a = np.array([20,30,40,50])
b = np.arange(4)
c = a - b
print(c,end='\n\n')
b**2
print(b,end='\n\n')
print(10*np.sin(a))
print(a<35)

#a = np.array([[1,1],[0,1]])
#b = np.array([[2,0],[3,4]])
##element by element multiplication
#print(a*b,end='\n\n')
##matrix multiplication
#print(a.dot(b))
##modify existing array:
#a+=a
#print(a,end='\n\n')
##Unary operations:
#print(a.sum())
#print(a.max(),a.min())
#    #Operations on axis
#b = np.arange(12).reshape(3,4)
#print(b,end='\n\n')
#print(np.shape(b))
#print(b.sum(axis=0))
#print(np.shape(b))
#print(len(b[0]))
#c = np.arange(10)
#print(c)
#print(np.shape(c))
#print(len(c))

a = np.arange(12).reshape(4,3)
print(a)
print(np.shape(a))
#d0 = a.sum(axis = 0)
#d1 = a.sum(axis = 1)
#print(d0)
#print(d1)
b = np.arange(3)
print(np.shape(b))
print(b)
