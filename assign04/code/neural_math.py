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
    return 1/(1+ math.exp(-7*x))

#Derivative of activation function
def sigmoid_derivative(value):
    return value*(1.0-value)

def get_deviation(sigma, iteration, max_iterations):
    return(sigma*math.exp((-iteration*math.log(sigma))/max_iterations))

def get_learn_rate(iteration, max_iterations):
    return math.exp((-iteration)/max_iterations)

def gauss_kern(x, sigma):
    return math.exp(x/((-2.0)*(sigma**2)))

