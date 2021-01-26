import random
import math

#Gaussian nonzero random normalized value, above 0
def randomNormalValue(mean,sigma):
    x = random.normalvariate(mean,sigma)
    while x==0 or abs(x)>1:
        x = random.normalvariate(mean,sigma)
    return x
#Uniform random value between 0 and 1
def randomValue():
    return (random.randint(1,100)/100.0)

#Return sigmoid activation function
def sigmoid(x):
    return 1/(1+ math.e**(-7*x))

#Derivative of activation function
def sigmoid_derivative(value):
    return value*(1.0-value)

def get_deviation(standard_deviation, iteration, max_iterations):
    variable = max_iterations/math.log(standard_deviation)
    return(standard_deviation*math.exp((-iteration)/variable))

def get_neighbourhood(n_init, iteration, max_iterations):
    return n_init*math.exp((-iteration)/max_iterations)

