###Library for minimization technique, 
###likelihood and Loglikelihood, examples of fitting with minuit library
###minimization, bisection and golden ratio methods

import numpy as np
import matplotlib.pyplot as plt
import math 
from math import pow,log,ceil,floor
import random
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


###example of Fitting with LeastSquares using iminuit###
###tipically used for regression (xi,yi) yi = f(xi,parameters) + sigmai

def model_function(x):
    return 

least_squares = LeastSquares(x_coord, y_coord, sigma_y, model_function)
my_minuit = Minuit(least_squares, m = 0, q = 0) ###m=0 q=0 starting point for the parameter phase space
my_minuit.migrad() ###find minimum
my_minuit.hesse()###calculates uncertainties cov matrix
my_minuit.minos() ###provides accurate parameter's value and uncertainties with simplex method

# printing formatted results for checking fit quality Q2,dof,pvalue
for par, val, err in zip(my_minuit.parameters, my_minuit.values, my_minuit.errors):
    print(f'{par} = {val:.3f} ± {err:.3f}') 

print(f'Goodness of the fit: {my_minuit.valid}')
print(f'Minuit Q2: {my_minuit.fval}')
print (f'Associated p-value: {1. - chi2.cdf(my_minuit.fval, df = my_minuit.ndof)}')

###other info 
print(my_minuit.covariance)
print(my_minuit.covariance['m','q'])
print(my_minuit.covariance.correlation())
display(my_minuit) ###displays all info

###also usually with data like this its usefull to scatter plot
sigma_y = epsilon_sigma * np.ones(len(y_coord))
fig,ax = plt.subplots()
ax.errorbar(x_coord,y_coord,xerr= , yerr=sigma_y, linestyle = 'None',marker='o')
ax.set_title()
ax.set_xlabel()
ax.set_ylabel()
ax.grid()
plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Fit for binned distribution, using binned likelihood###

###the model need to be the cdf of the binned function in analysis
def model (bin_edges,N_signal,mu,sigma,N_backgroud,tau):
    return N_signal * norm.cdf(bin_edges,mu,sigma) + N_background*expon.cdf(bin_edges,0,tau)


from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
my_cost_func = ExtendedBinnedNLL(bin_content, bin_edges, model)
####ExtendedBinnedNLL extends the BinnedNLL by allowing for an additional parameter representing the expected number of background events in each bin.###
#This additional parameter accounts for background contributions or other external sources of events that are not explicitly modeled by the fit.
###give clues for parameters helping minuit
my_minuit = Minuit(my_cost_func,N_signal = N_evnt, mu = samle_mean, sigma = sample_sigma, N_backgroud=N_evnt,tau= )
my_minuit.limits['','',''] = (0,None) ##this bounds parameters to be positive/negative
my_minuit.migrad()
my_minuit.minos()
print(my_minuit.valid)
display(my_minuit)
print(my_minuit.params) ##prints parameters and unc
print(my_minuit.covariance)###prints covariance matrix
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#Example of helping minuit with a partial fit

 
# setting the signal to zero for a first background-only preliminary fit
my_minuit.values["N_signal"] = 0
# fixing the value of the signal parameters to a constant value
# (i.e. they are not considered in the fit)
my_minuit.fixed["N_signal", "mu", "sigma"] = True

# temporary mask avoiding the signal region
bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
my_cost_func.mask = (bin_centres < 5) | (15 < bin_centres)

my_minuit.migrad ()

#second partial fit performed with background frozen 
my_cost_func.mask = None # remove mask
my_minuit.fixed = False # release all parameters
my_minuit.fixed["N_background", "tau"] = True # fix background parameters
my_minuit.values["N_signal"] = N_events - my_minuit.values["N_background"] # do not start at the limit
my_minuit.migrad ()

#final fit 
my_minuit.fixed = False # release all parameters
my_minuit.migrad ()
print (my_minuit.valid)
display (my_minuit)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###fit with MaxLL for unbinned distribution###
###the model needs to be the pdf itself###
def model(x,parameters):
    return function.pdf(x,parameters)

from iminuit import Minuit 
from iminuit.cost import UnbinnedNLL
my_cost_func = UnbinnedNLL(sample,model)
###then the fitting function are the same

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Applying least squares Q2 for histogram fit, the model scales the pdf by the N events###
def model(x,N_evnt,mu,sigma,bin_width):
    return norm.pdf(x,mu,sigma)*bin_width*N_evnt

from iminuit import Minuit
from iminuit.cost import LeastSquares
least_squares = LeastSquares(bin_center,bin_content,sigma_y,mu,model)
###then the fitting is the same for the binned maxlikelihood


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
####normal binned fitting with NLL###
from iminuit import Minuit
from iminuit.cost import BinnedNLL
from Ipython.display import display

bin_content, bin_edges = np.histogram(sample, bins = floor(N_event/100), range = (floor(min(sample)), ceil(max(sample))))

def model_function(bin_edges,mu,sigma):
    return norm.cdf(bn_edges,mu,sigma)

my_cost_func = BinnedNLL(bin_content, bin_edges, model_function)
my_minuit = Minuit(my_cost_func, mu = sample_mean, sigma =std_mean )
my_minuit.migrad()
my_minuit.minos()
display(my_minuit)
print(my_minuit.valid)
print (f'Associated p-value: ', 1.-chi2.cdf(my_minuit.fval,df=my_minuit.dof))
if 1.-chi2.cdf(my_minuit.fval,df=my_minuit.dof) > 0.10 :
    print('The model passed the test')

###ExtendedBinneNLL, fit binned with Maximum Likelihood

bin_content,bin_edges = np.histogram(sample, bins = floor(N_event/100), range = (floor(min(sample)),ceil(max(sample))))
def model_binned (bin_edges, mu, sigma):
    return norm.cdf(bin_edges,mu,sigma) * N_signal

my_cost_func = ExtendedBinnedNLL(bin_content, bin_edges, model_binned)
my_minuit = Minuit(my_cost_func, N_signal= , mu= , sigma= )
my_minuit.migrad()
assert my_minut.valid
display(my_minuit)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Calculate Q2 value###
# epsilon_sigma = c
# epsilon = generate(xMin,xMax,epsilon_sigma)
# x_coord = np.arange(0,10,1)
# y_coord = np.zeros(10)
# for i in range(x_coord.size):
#     y_coord[i] = f(x_coord[i],m,q) + epsilon[i])
# sigma_y = epsilon_sigma*np.ones(len(y_coord)

#def calc_Q2():
#    Q2 = 0.
#   for x,y,ery in zip(x_coord,y_coord,sigma_y):
#       Q2= Q2 + pow( y - func(x,m,q))/ery, 2)
#    return Q2   


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###MINIMIZATION FINDING ZEROS and MAX/MIN

## finding minimum (root) of a function f(x) using bisect method with scipy ##

from scipy.optimize import bisect

def f(x):
    return x**2 - 1
root = bisect(f, 0, 3)

print(f"Root of the function is: {root:.2f}")


###Calculate zero of a f(x) using bisection algorithm, in a definite interval###
def bisection(f, xMin: float, xMax: float, prec: float = 0.0001) -> float:
    if f(xMin) * f(xMax) >= 0.:
        raise ValueError('xMin and xMax such that: f(xMin)*f(xMax) < 0 ')
    
    xAve = xMin
    while (xMax - xMin) > prec:
        xAve = (xMax + xMin) * 0.5
        if f(xAve) * f(xMin) > 0.:
            xMin = xAve
        else:
            xMax = xAve
    return xAve
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Calculate the same but with a recursive method###
def bisection_ricorsiva(f,xMin: float, xMax: float, prec: float=0.0001) -> float:
    
    if f(xMin)*f(xMax) >= 0.:
        raise ValueError('xMin and xMax such that: f(xMin)*f(xMax) < 0 ')
    xAve = 0.5 * (xMax + xMin)
    
    if ( (xMax-xMin) < prec): return xAve;
    if (f(xAve) * f(xMin) >0.) : return bisezione_ricorsiva(f,xAve,xMax,prec);
    else                       : return bisezione_ricorsiva(f,xMin,xAve,prec);


###Calculates the zero of a f(x) using bisection method but also returns the list of intervals used for the algorithm###
def bisezione_intervals(f,xMin: float, xMax: float, prec: float=0.0001) ->float,list[float]:
    if f(xMin)*f(xMax) >= 0.:
        raise ValueError('xMin and xMax such that: f(xMin)*f(xMax) < 0 ')
    intervals = []
    xAve = xMin
    while ((xMax-xMin)>prec):
        xAve = (xMax+xMin) * 0.5
        if ( f(xAve) * f(xAve) > 0.): xMin = xAve
        else                        : xMax = xAve
    return xAve,intervals

                               
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Calculate the min  of f(x) using the Golden Ratio method, note that f(x) has to have only one minimun in the given interval###
def golden_ratio_min(f, x0: float, x1: float, prec: float=0.0001) ->float:
    if x0 > x1:
        raise ValueError('xMin has to be smaller than xMax')

    r = 0.618
    x2 = 0.
    x3 = 0.
    width = abs(x1 - x0)
    while ( width > prec):
        x2 = x0 + r*(x1-x0)
        x3 = x0 + r*(1.-r)*(x1-x0)
        if( f(x3) > f(x2) ):
            x0 = x3
            x1 = x1
        else: 
            x1 = x2
            x0 = x0
        width = abs(x1-x0)
    return (x0+x1) / 2.

def golden_ratio_min(f, x0: float, x1: float, prec: float = 0.0001) -> float:
    if x0 > x1:
        raise ValueError('x0 has to be smaller than x1')
    # Golden ratio constant
    golden_ratio = 0.618
    # Initialize x2 and x3
    x2 = x0 + golden_ratio * (x1 - x0)
    x3 = x0 + golden_ratio * (1. - golden_ratio) * (x1 - x0)
    # Initial width of the interval
    width = abs(x1 - x0)
    # Iterate until the width is smaller than the precision
    while width > prec:
        if f(x3) > f(x2):
            # Update x0 to x3
            x0 = x3
            # Update x2 to x3
            x2 = x3
            # Update x3 to a new value
            x3 = x0 + golden_ratio * (1. - golden_ratio) * (x1 - x0)
        else:
            # Update x1 to x2
            x1 = x2
            # Update x2 to x3
            x2 = x3
            # Update x3 to a new value
            x3 = x0 + golden_ratio * (x1 - x0)

        # Update the width of the interval
        width = abs(x1 - x0)
    # Return the midpoint of the final interval as the approximate minimum
    return (x0 + x1) / 2

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Calculate the max of f(x) using Golden Ratio method###
def golden_ratio_max(f, x0: float, x1: float, prec: float=0.0001) ->float:
    if x0 > x1:
        raise ValueError('xMin has to be smaller than xMax')

    r = 0.618
    x2 = 0.
    x3 = 0.
    width = abs(x1 - x0)
    while ( width > prec):
        x2 = x0 + r*(x1-x0)
        x3 = x0 + r*(1.-r)*(x1-x0)
        if( f(x3) < f(x2) ):
            x0 = x3
            x1 = x1
        else: 
            x1 = x2
            x0 = x0
        width = abs(x1-x0)
    return (x0+x1) / 2.

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Another implementation of the same algorith###
def golden_ratio_max(f, x0: float, x1: float, prec: float = 0.0001) -> float:
    if x0 > x1:
        raise ValueError('x0 has to be smaller than x1')

    golden_ratio = 0.618 
    x2 = x0 + golden_ratio * (x1 - x0)
    x3 = x0 + golden_ratio * (1. - golden_ratio) * (x1 - x0)
    width = abs(x1 - x0)
    while width > prec:
        if f(x3) < f(x2):  # Compare function values to find the maximum
            x0 = x3
            x2 = x3
            x3 = x0 + golden_ratio * (1. - golden_ratio) * (x1 - x0)
        else:
            x1 = x2
            x2 = x3
            x3 = x0 + golden_ratio * (x1 - x0)
        width = abs(x1 - x0)
    return (x0 + x1) / 2

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Calculate the max of f(x) using Golden Ratio method, ricorsivamente###
def golden_ratio_min_ricorsiva(g,x0,x1,prec=0.0001):
    r= 0.618
    x2 = 0.
    x3 = 0.
    larghezza = abs(x1-x0)

    while (larghezza > prec):
        x2= x0 + r*(x1-x0)
        x3= x0 +(1.-r)*(x1-x0)

        if(larghezza<prec)   : return (x0+x1) /2
        elif (g(x3) > g(x2)) : return golden_ratio_min_ricorsiva(g,x3,x1,prec)
        else                 : return golden_ratio_min_ricorsiva(g,x0,x1,prec)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
def golden_ratio_max_ricorsiva(g,x0,x1,prec=0.0001):
    r = 0.618
    x2 = 0.
    x3 = 0.
    larghezza = abs(x1-x0)

    if (larghezza < prec) : return (x0+x1)/2
    elif (g(x3) < g(x2))  : return golden_ratio_max_ricorsiva(g,x3,x1,prec)
    else                  : return golden_ratio_max_ricorsiva(g,x0,x1,prec)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Likelihood and logLikelihood###

###Calculate the likelihood with input a sample of data, the pdf of the sample pdf(x,parameter)###
def likelihood(sample: list[float], parameter: float, pdf) -> float:
    L = 1.0 
    for x in sample:
        L *= pdf(x, parameter)
    return L

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Calculate the logLikelihood for a sample with a pdf(x,parameter)###
def log_likelihood(sample: list[float], parameter:float, pdf)->float:
    logL = 0.
    for x in sample:
        if( pdf(x,parameter) > 0. ): logL+= log(pdf(x,parameter))
    return logL



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Parameters estimate using Maximum likelihood###


###Golden ratio method used to find the max of logLikelihood)###
###here f is the likelihood function to wich we'll find the maximum##
###pdf is the prob density funtion of the sample###

def golden_ratio_logL(sample: list[float], x0: float, x1: float, f, pdf, prec: float = 0.0001) -> float:
    if x0 > x1: 
        raise ValueError('x0 has to be smaller than x1')

    r = 0.618
    x2 = 0.
    x3 = 0.
    width = abs(x1 - x0)
    while width > prec:  # Corrected the condition
        x2 = x0 + r * (x1 - x0)
        x3 = x0 + r * (1. - r) * (x1 - x0)
        if f(sample, x3, pdf) < f(sample, x2, pdf):  # f here is the loglikelihood(sample, parameter, pdf)
            x0 = x3
            x1 = x1 ###this may be useless
        else:
            x1 = x2
            x0 = x0 ##this may be useless###
        width = abs(x1 - x0)
    return (x0 + x1) / 2
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###To find the uncertainties of the loglikelihood estimates, we use the bisection method###
###we have a definite interval xMin and xMax, yLevel is the value of the horizontal intersect, 
# theta_hat is the maximum of the likelihood, 
# f is the function for which we find the zero and pdf is the pdf of the sample###

def bisection_logL(sample: list[float], xMin:float, xMax:float, yLevel: float, theta_hat: float, f, pdf, prec: float=0.0001) -> float:
    def f_1 (x):
        return f(x,pdf,sample,theta_hat) - yLevel

    xAve = xMin
    while (xMax-xMin) > prec:
        xAve = 0.5*(xMax + xMin)
        if f_1(xAve) * f_1(xMin) >0.:
            xMin = xAve
        else: 
            xMax = xAve
    return xAve


def bisection_logL(sample: list[float], xMin: float, xMax: float, yLevel: float, theta_hat: float, f, pdf) -> (float, float):
    def f_1(x):
        return f(x, theta_hat, pdf, sample) - yLevel

    xAve = None
    while (xMax - xMin) > 0.0001:  # Adjust the threshold if needed
        xAve = 0.5 * (xMax + xMin)
        if f_1(xAve) * f_1(xMin) > 0.:
            xMin = xAve
        else:
            xMax = xAve
    return xMin, xMax
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###implementing in a program, to find the confidence intervall###
tau_hat = golden_ratio_logL(sample,x0,x1,loglikelihood,exp_pdf,prec=0.00001)

tau_hat_minus=bisection_logL(sample,0.5,theta_hat,-0.5,theta_hat,loglikelihood_ratio, exp_pdf,prec=0.0001)

tau_hat_plus=bisection_logL(sample,theta_hat,5,-0.5,theta_hat,loglikelihood_ratio, exp_pdf,prec=0.0001)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
####how to find uncertainties with bisection method, finding the intersection points of the LogLikelihood function

def log_likelihood(sample: list[float], parameter: float, pdf) -> float:
    logL = 0.
    for x in sample:
        pdf_val = pdf(x, parameter)
        if pdf_val > 0.:
            logL += math.log(pdf_val)
    return logL

def find_uncertainties(sample: list[float], theta_hat: float, pdf, yLevel: float, xMin: float, xMax: float, prec: float = 0.0001) -> float:
    """
    Finds the uncertainties of the maximum log-likelihood method for estimating a parameter.

    Args:
    - sample: A list of observed data points.
    - theta_hat: The estimated parameter value.
    - pdf: The probability density function of the sample.
    - yLevel: The value of the horizontal intersect.
    - xMin: The lower bound of the interval.
    - xMax: The upper bound of the interval.
    - prec: The precision for the bisection method (optional).

    Returns:
    - uncertainty: The uncertainty in the parameter estimate.
    """
    def f_1(x):
        return log_likelihood(sample, x, pdf) - yLevel

    # Use bisection method to find the intersection point
    xMin_intersection, xMax_intersection = bisection_logL(sample, xMin, xMax, yLevel, theta_hat, f_1, pdf)

    # Calculate uncertainty as the difference between the intersection points
    uncertainty = abs(xMax_intersection - xMin_intersection)

    return uncertainty

def bisection_logL(sample: list[float], xMin: float, xMax: float, yLevel: float, theta_hat: float, f, pdf) -> (float, float):
    """
    Bisection method to find the intersection point of the log-likelihood function with a horizontal line.

    Args:
    - sample: A list of observed data points.
    - xMin: The lower bound of the interval.
    - xMax: The upper bound of the interval.
    - yLevel: The value of the horizontal intersect.
    - theta_hat: The estimated parameter value.
    - f: The function for which to find the zero.
    - pdf: The probability density function of the sample.

    Returns:
    - xMin_intersection: The lower bound of the interval containing the intersection point.
    - xMax_intersection: The upper bound of the interval containing the intersection point.
    """
    xAve = None
    while (xMax - xMin) > 0.0001:  # Adjust the threshold if needed
        xAve = 0.5 * (xMax + xMin)
        if f(xAve) * f(xMin) > 0.:
            xMin = xAve
        else:
            xMax = xAve
    return xMin, xMax

    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
#Generating a Toy Experiment plus Maximum Likelihood

single_toy = np.random.expon(lamb= lambda_value, size = sample_size)


tau_hats = []
for _ in range(N_toys): 
    single_toy = np.random.expon(lamb= lambda_value, size = sample_size)
    tau_hat_toy = golden_ratio_logL(single_toy, x0, x1, loglikelihood, exp_pdf, prec=0.00001)
    tau_hats.append(tau_hat_toy)
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
#libreria di funzioni per fitting dati (da git)


from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL
from scipy.stats import chi2

def LS_fit(data_x:list, data_y:list, y_err:list, model:'function', disp = 1, **kwrds):
    """
    Fit dei dati con metodo dei minimi quadrati.

    Ritorna in ordine: parametri, valori, errori, p-value, gradi di lib., chi-quadro, matrice di covarianza
    
    Usare x, y, *z = LS_fit(...), dove *z racchiude tutte gli altri valori
    """
    cost_function = LeastSquares(data_x, data_y, y_err, model)

    my_minuit = Minuit(cost_function, **kwrds)
    my_minuit.migrad()
    my_minuit.hesse()

    params = my_minuit.parameters
    values = my_minuit.values
    uncert = my_minuit.errors
    chi_quadro = my_minuit.fval
    dof = my_minuit.ndof
    cov = my_minuit.covariance

    pval = 1. - chi2.cdf(chi_quadro, df = dof)

    if disp : display(my_minuit) # type: ignore

    return params, values, uncert, pval, dof, chi_quadro, cov


def Binned_fit(bin_content:list, bin_edges:list, modello:'function', disp = 1, **kwrds):
    """
    Fit di dati da un istogramma.

    Ritorna in ordine: parametri, valori, errori, p-value, gradi di lib., chi-quadro, matrice di covarianza
    """
    cost_function = ExtendedBinnedNLL(bin_content, bin_edges, modello)

    my_minuit = Minuit(cost_function, **kwrds)
    my_minuit.migrad()
    my_minuit.hesse()

    params = my_minuit.parameters
    values = my_minuit.values
    uncert = my_minuit.errors
    chi_quadro = my_minuit.fval
    dof = my_minuit.ndof
    cov = my_minuit.covariance

    pval = 1. - chi2.cdf(chi_quadro, df = dof)

    if disp : display(my_minuit) # type: ignore

    return params, values, uncert, pval, dof, chi_quadro, cov

def TestCompatibilita(x0:float, sigma0:float, x1:float, sigma1:float = 0) -> float:
    """
    Test di compatibilità tra due valori.
    """
    sigma = (sigma0**2 + sigma1**2)**0.5
    z = abs(x0 - x1) / sigma

    return z
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#example of toy experiment with fit Q2 and p-value 
def gen_y(x_values,sigma,p0,p1):
    return model(x_values,p0,p1) + np.random.normal(0,sigma,len(x_values))




N_toys = 1000
Q_2_values = []
p_values = []

for i in range(N_toys):
    y_toy = gen_y(x_values,sigma,2,3)
    lest_squares_toy = LeastSquares(x_values, y_toy, sigma, model)
    m_toy = Minuit(lest_squares_toy, p0 =2 , p1=3)
    m_toy.migrad()
    m_toy.hesse()
    m_toy.minos()
    Q_2_values.append(m_toy.fval)
    p_values.append(1. - chi2.cdf(m_toy.fval, df = m_toy.ndof))


plt.hist(Q_2_values, bins=30, alpha=0.7, label='$Q^2$ values')
plt.hist(p_values, bins=30, alpha=0.7, label='$p$ values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('toy_experiments.png')
plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# EXAMPLE OF BINNED fitting a gaussian to a generated sample 
from iminuit import Minuit
from iminuit.cost import BinnedNLL
from scipy.stats import norm, chi2, expon


def model_cdf(bin_edges, mu, sigma):
    return norm.cdf(bin_edges,mu,sigma)

bin_content, bin_edges = np.histogram(sample_clt, bins = N_bins, range=(xMin,xMax))


least_squares = BinnedNLL(bin_content, bin_edges, model_cdf)
m = Minuit(least_squares, mu = 1 , sigma=1)
m.migrad()
m.hesse()
m.minos()
display(m)


for key in m.parameters:
    print(f'{key} = {m.values[key]:.3f} ± {m.errors[key]:.3f}')
    
p_value = 1. - chi2.cdf(m.fval, df = m.ndof)
print(f'Goodness of the fit: {m.valid}')
print(f"p-value: {p_value:.3f}")  
if p_value < 0.05:
    print("The fit is not good")
else:
    print("The fit is good")


    




