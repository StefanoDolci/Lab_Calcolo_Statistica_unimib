###Library for generating pseudo random number###

 
import math
import numpy as np
import matplotlib.pyplot as plt
from stats import stats
import statistics as stats
from math import ceil,floor,pow 
import random
from random import uniform

###Generate N pseudo random number distributed between 0 and 1
def gen_uniform(N: int, seed:float=0.)->float:
    if seed!=0. : random.seed(flaot(seed))
    randlist= []
    for i in range (N):
        randlist.append(random.random())
    return randlist
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate one random number distributed uniformly between given range###
def uniform_range(xMin: float, xMax: float)->float:
    return xMin + random.random() * (xMax-xMin)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate a list of N random number distr uniformly between given range###
def uniform_range_list(xMin: float, xMax: float,N: int, seed: float=0.)-> list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist = []
    for i in range(N):
        randlist.append(uniform_range(xMin,xMax))
    return randlist
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate one random number with TAC metohd###
def TAC_range(f, xMin: float, xMax: float,yMax: float, seed:float=0.):
    if seed!=0. : random.seed(float(seed))
    x = random.uniform(xMin,xMax)
    y = random.uniform(0,yMax)
    while ( y > f(x) ):
        x = random.uniform(xMin,xMax)
        y = random.uniform(0,yMax)
    return x
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate a list of N random number according to the TAC method###
def TAC_range_list(f,xMin: float, xMax: float, yMax: float,N: int,seed: float = 0.)-> list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist = []
    for i in range(N):
        x = random.uniform(xMin,xMax)
        y = random.uniform(0,yMax)
        while ( y > f(x) ):
            x = random.uniform(xMin,xMax)
            y = random.uniform(0,yMax)
        randlist.append(x)
    return randlist
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Generate a random number according to the CLT on a given interval###
def CLT_range(xMin: float,xMax: float,N_sum: int=100) ->float:
    y = 0.
    for i in range(N_sum):
        y = y+ uniform_range(xMin,xMax)
    y /= N_sum
    return y 

def CLT_range_2(xMin: float, xMax: float, N_sum: int=100,seed: float=0.) ->float:
    if seed!=0. : random.seed(float(seed))
    return np.sum(uniform_range_list(xMin,xMax,N_sum,seed))/N_sum

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate a list on N number according to CLT on a given interval###

def CLT_range_list(xMin: float, xMax: float,N: int, N_sum: int=100,seed: float=0.) ->list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist = []
    for i in range(N):
        randlist.append(CLT_range(xMin,xMax,N_sum))
    return randlist

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Generate a random number according to the CLT , with mean and sigma###

def CLT_range_ms(mean: float,sigma: float,N_sum: int=100) ->float:
    y = 0.
    delta = np.sqrt(3*N_sum)*sigma
    xMin = mean-delta
    xMax = mean+delta
    for i in range(N_sum):
        y = y + uniform_range(xMin,xMax)
    y/=N_sum
    return y

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Generate N random number according to the CLT, with mean and sigma###

def CLT_range_list_ms(mean: float,sigma: float,N: int,N_sum: int=100,seed: float=0.) ->list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist = []
    delta = np.sqrt(3*N_sum)*sigma
    xMin = mean-delta
    xMax = mean+delta
    for i in range(N):
        randlist.append(CLT_range_ms(xMin,xMax,N_sum))
    return randlist
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generatae a single random number with CLT method on fixed interval,
def CLT_range_min_max(xMin: float, xMax: float, yMin= float,yMax=float,N_sum:int=10)->float:
    xMin = 
    xMax = 
    yMin = 
    yMax = 
    y = 0.
    for i in range(N_sum):
        y += pdf(xMin,xMax,yMin,yMax)
    y/=N_sum
    return y
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate a random number with inverse function algorithm###
### inverse_cdf: inverse pdf's cumulatve density function, used to gen number according to a certain pdf
def inv_generate(inverse_cdf, seed: float=0.) ->float:
    if seed!=0. : random.seed(float(seed))
    y = random.uniform(0,1)
    return inverse_cdf(y)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
def inv_generate_list(inverse_cdf,N: int, seed: float =0.) ->list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist= []
    for i in range(N):
        randlist.append(inv_generate(inverse_cdf,seed))
    return randlist

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate random number with the inverse function method using cdf of the exponential, remember that lambda = 1./tau
def inv_exp(tau: float, seed: float=0.) ->float:
    if seed!=0. : random.seed(float(seed))
    y = random.uniform(0,1)
    return -1*(np.log(1-y))*tau

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate a list of random number according to the inverse function algorithm following the exponential###
###the cdf is y = 1 - e^(-x/tau)
def inv_exp_list(tau: float, N:int, seed:float=0.)-> list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist = []
    for i in range(N):
        randlist.append(inv_exp(tau,seed))
    return randlist
    
###This does the same thing exept using np arrays ###
def inv_exp_list_np(tau: float, N: int, seed: float=0.) ->list[float]:
    array = np.array(uniform_range_list(0,1,N,seed))
    return list(-(np.log(1-array))*tau)
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate a list of random number following a Poissoniana###
###lambda is the poissonian expected value lambda = t/tau###
###tau exponential expected value###

def rand_poisson(lambda_value: float, seed: float=0.)->list[float]:
    if seed!=0. : random.seed(float(seed))
    tau = 1 
    randlist =[]
    for i in range(N):
        delta_time = 0
        count = 0
        while (delta_time <= lambda_value):
            delta_time +=inv_exp(tau)
            if delta_time <= lambda_value:
                count+=1
        randlist.append(count)
    return randlist
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    
##Generate a random number folowing Poissonian, from exponential pdf
def gen_poiss(mean):
    total_time = inv_exp(1.)
    events = 0.
    while (total_time<mean):
        events = events+1
        total_time = total_time + inv_exp(1.)
    return events
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Gen N random number following poissonian
def gen_poiss_list(mean,N,seed=0.):
    if seed!=0.: random.seed(float(seed))
    randlist=[]
    for i in range(N):
        randlist.append(gen_poiss(mean))
    return randlist
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Generate random numbers following a Poissonian with numpy###
def generate_poisson_random_numbers(lmbda, N=1):
    """
    Generate pseudo-random numbers following a Poisson distribution.

    Parameters:
    - lmbda (float): The rate parameter (λ) of the Poisson distribution.
    - N (int or tuple of ints, optional): Output shape. Default is 1.

    Returns:
    - numpy.ndarray or float: Random samples from the Poisson distribution.
    """
    return np.random.poisson(lmbda, N)



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Library for Integration methods 


#Calc of a definite integral of function f with hit or miss method, f must be positive and continuos in the definite interval###
def HOM_integration (f, xMin: float, xMax: float, yMax: float, N: int=10000, seed: float=0.) ->tuple[float,float]:
    if xMin>xMax:
        raise ValueError('Error: xMin has to be smaller than xMax')
    if xMin == xMax:
        return 0.,0.
    if seed!=0.: 
        random.seed(float(seed))
        
    x_coord = uniform_range_list(xMin,xMax,N)
    y_coord = uniform_range_list(0,yMax,N)
    points_under = 0.
    
    for x,y in zip(x_coord,y_coord):
        if ( f(x) >= y):
            points_under+=1

    area_rett = (xMax-xMin) * yMax
    fraction = float(points_under)/ float (N)
    integral_value = area_rett * fraction
    integral_uncertainty = area_rett**2 * fraction * (1-fraction) / N
    return integral_value,math.sqrt(integral_uncertainty)
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Calc of a definite integral of a function f with the crude MonteCarlo method###
### using the properties of the Expectation value of a function and uniform distribution###
###we can estimate the integral with it's uncertainty###
###Note N here is the number of generated xi unif distributed###
def CRUDE_MC_integration(f, xMin: float, xMax: float, N: int = 10000, seed: float = 0.) -> tuple[float, float]:
    if xMin >= xMax:
        raise ValueError('Error: xMin has to be smaller than xMax')
    if xMin == xMax:
        return 0., 0.
    if seed != 0.:
        random.seed(float(seed))

    summ = 0.
    summ_sq = 0.
    for i in range(N):
        x = uniform_range(xMin, xMax) 
        summ += f(x)
        summ_sq += f(x) * f(x)

    mean = summ / float(N)
    variance = summ_sq / float(N) - mean * mean
    variance = variance * (N - 1) / N
    # Calculate the length of the interval
    length = xMax - xMin
    # Calculate the estimated integral and its uncertainty
    integral_estimate = mean * length
    integral_uncertainty = math.sqrt(variance / N) * length

    return integral_estimate, integral_uncertainty



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Another implementation of the same algorithm###
def CRUDE_MC_integration(f, xMin: float, xMax: float, N_event: int = 10000, seed: float = 0.) -> tuple[float, float]:
    if xMin >= xMax:
        raise ValueError('Error: xMin has to be smaller than xMax')
    if xMin == xMax:
        return 0., 0.
    if seed != 0.:
        random.seed(float(seed))

    x = np.random.uniform(xMin, xMax, N_event) 
    y = np.array(list(map(f, x)))  # Apply the function f to each element of x and convert to numpy array

    mean = np.mean(y)
    variance = np.var(y)
    integral = (xMax - xMin) * mean
    integral_unc = np.sqrt(variance / N_event)
    
    return integral, integral_unc

    
    
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##Example of generating and plotting pseudo random numbers following a certain pdf##
def TAC_range_list(f,xMin: float, xMax: float, yMax: float,N: int,seed: float = 0.)-> list[float]:
    if seed!=0. : random.seed(float(seed))
    randlist = []
    for i in range(N):
        x = random.uniform(xMin,xMax)
        y = random.uniform(0,yMax)
        while ( y > f(x) ):
            x = random.uniform(xMin,xMax)
            y = random.uniform(0,yMax)
        randlist.append(x)
    return randlist


sample = TAC_range_list(normalized_pdf,0,1.5*np.pi,1,10000)

N_bins = sturges(sample)
xMin= floor(min(sample))
xMax= ceil(max(sample))
bin_edges = np.linspace(xMin,xMax,N_bins)

fig, ax = plt.subplots(nrows=1,ncols=1)
plt.hist(sample,bins=bin_edges,color="orange",label="TAC method generated sample")
plt.title("Histogram of normalized pdf generated by TAC method")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.grid()
plt.show()
    

    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#Example of toys experiment 
N_toys = 10000

Q_2_values = []

for i in range(N_toys):
    x_coord = uniform_range_list(0,10,10)
    x_coord.sort()
    y_coord = np.zeros(len(x_coord))
    for i in range(len(x_coord)):
        y_coord[i]= func(x_coord[i],a,b,c) + np.random.normal(0,10)
    least_squares = LeastSquares(x_coord,y_coord,10,func)
    m = Minuit(least_squares, a = 3, b = 2, c = 1)
    m.migrad()
    Q_2_values.append(m.fval)

plt.hist(Q_2_values, bins=30, alpha=0.7, label='$Q^2$ values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
