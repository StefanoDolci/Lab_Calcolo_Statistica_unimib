###Library for statistis and plotting 

import numpy as np 
import math
from math import floor,ceil,pow,log,sqrt
import matplotlib.pyplot as plt





###sturges rule
def sturges(sample:list)->int:
    return int(np.ceil(1+3.322 * np.log(len(sample))))

###sturges rule with a number as input###
def sturges(N: int)->int:
    return int(np.ceil(1 + 3.222 * np.log(N)))
               
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

###Stats of a sample

def mean(sample: list[float]) ->float:
    return sum(sample)/len(sample)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def variance(sample: list[float], bessel: bool= True) ->float:
    summ = 0.
    summ_sq = 0.
    N = len(sample)
    for elem in sample:
        summ+=elem
        summ_sq+=elem*elem
    var = summ_sq / N - summ*summ / (N*N)
    if bessel:
        var = N *var / (N-1)
    return var

def variance(sample: list[float], bessel: bool=True) -> float:
    mean_sq = mean(list(map(lambda x: x**2,sample)))
    squared_mean = mean(sample)**2
    var = mean_sq - squared_mean
    if bessel is True:
        N = len(sample)
        var = N * var/ (N - 1)
    return var
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def std(sample: list[float],bessel: bool=True) ->float:
    return sqrt(variance(sample,bessel))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
def std_mean(sample: list[float],bessel: bool=True) ->float:
    return sqrt(variance(sample,bessel)/len(sample))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def skewness(sample: list[float],bessel: bool=True) ->float:
    mean_sample = mean(sample)
    skew = 0.
    for x in sample:
        skew = skew +math.pow(x-mean_sample,3)
    skew = skew / ( len(sample)*math.pow(std(sample,bessel),3))
    return skew 
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def kurtosis(sample: list[float],bessel: bool=True) ->float:
    mean_sample = mean(sample)
    kurt = 0.
    for x in sample:
        kurt = kurt + math.pow(x-mean_sample,4)
    kurt = kurt / ( len(sample)*math.pow(variance(sample,bessel),2))-3
    return kurt 
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##Statistics but with a class###

class stats: 
    summ = 0.
    sumSq = 0.
    N = 0
    sample = []
    
    ###Read input list of data###
    def __init__(self,sample):
        self.sample = sample
        self.summ = sum (self.sample)
        self.sumSq = sum( [x*x for x in self.sample])
        self.N = len(self.sample)
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    def mean(self):
        return self.summ/self.N

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    
    def variance(self, bessel = True):
        var = self.sumSq / self.N - self.mean() * self.mean()
        if bessel : var = self.N * var / (self.N-1)
        return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    
    def sigma(self,bessel = True):
        return sqrt(self.variance(bessel))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    
    def sigma_mean(self,bessel = True):
        return sqrt(self.variance(bessel)/self.N)
        
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    
    def kurtosis(self):
        mean = self.mean
        kurt= 0.
        for x in self.sample:
            kurt = kurt + pow(x-mean,4)
        kurt = kurt / (self.N * pow (self.variance(),2)) -3
        return kurt 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
    def skewness(self):
        mean = self.mean()
        asymm= 0.
        for x in self.sample:
            asymm = asymm + pow(x-mean, 3)
        asymm = asymm / (self.N * pow (self.sigma(),3))
        return asymm
            
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###add an element al sample##
    def append(self,x):
        self.sample.append(x)
        self.summ = self.summ + x
        self.sumSq= self.sumSq + x*x
        self.N = self.N + 1
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Also implementing scipy.stats can be usefull 

from scipy.stats import norm 
norm.pdf(x,mean,sigma)
norm.cdf(x,mean,sigma)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###calculating first four moments of a pdf

mu,var,skew,kurt = norm.stats(moments = 'mvsk')
print(mu,var,skew,kurt)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###Data visualization histograms with options

def histogram (sample: list[float], 
               title: str = 'Histogram', 
               xlabel: str= 'X axis', 
               ylabel: str= 'Y axis', 
               label: str= 'histogram',
               color: str= 'green', 
               sturges: bool=True):
    
    N_bins =  sturges(dati)
    xMin= floor(min(dati))
    xMax=ceil(max(dati))
    bin_edges = np.linspace(xMin,xMax,N_bins)
    
    if sturges is True:
        fig,ax = plt.subplots(nrows=1,ncols=1)
        ax.hist(sample, bins = bin_edges, label=label, color=color)  ###also for ex: ,histtype = 'step'####
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        plt.savefig('Histogram.png')
        plt.show()
        return 
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
####histogram with sample moments visualization###
def histogram_stats (sample: list[float], 
               title: str = 'Histogram', 
               xlabel: str= 'X axis', 
               ylabel: str= 'Y axis', 
               label: str= 'f(x)',
               color: str= 'green',
               sturges: bool=True):
    
    sample_mean = mean(sample);
    sample_sigma = std(sample);
    xMin = floor(min(sample))
    xMax = ceil(max(sample))
    
    if sturges is True:
        N_bins = sturges(sample)
        bin_edges = np.linspace(xMin,xMax,N_bins)
        fig,ax = plt.subplots()
        ax.hist(sample, bins = bin_edges, label=label, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        vertical_limits = ax.get_ylim()
        ax.plot([sample_mean,sample_mean],vertical_limits,color = 'blue')
        ax.plot([sample_mean-sample_sigma,sample_mean-sample_sigma],vertical_limits,color = 'blue', linestyle = 'dashed')
        ax.plot([sample_mean+sample_sigma,sample_means+sample_sigma],vertical_limits,color = 'blue', linestyle = 'dashed')
        ax.legend()
        ax.grid(True)
        plt.savefig('HistWithStats.png')
        plt.show()
        return 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

###plot a function with options
def plot(f,
         xMin: float,
         xMax: float,
         title: str='Plot',
         xlabel: str='X axis',
         ylabel: str='Y axis',
         label: str='plot',
         color: str='orange',
         log_scale: bool=False):
    
    x_coord = np.linspace(xMin,xMax,10000)
    y_coord = list(map(lambda x: f(x),x_coord))
    
    fig,ax= plt.subplot(nrows=1,ncols=1)
    ax.plot(x_coord,y_coord,label=label,color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    if log_scale == True:
        ax.set_xscale('log')
    plt.show()
    return 
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###scatter plot with error bars and options

def scatter(x_coord: list[float],
            y_coord: list[float],
            x_error: list[float],
            y_error: list[float],
            title: str= 'Scatter plot',
            xlabel: str= 'X axis',
            ylabel: str= 'Y axis',
            color: str= 'blue',
            label: str= 'scatter',
            marker: str= '.',
            log_scale: bool=False):
    
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.errorbar(x_coord,y_coord,xerr=x_error,yerr=y_error,label=label,color=color,marker=marker)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    if log_scale == True:
        ax.set_xscale('log')
    plt.show()
    return
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Build a numpy histogram containing data
N_bins= floor(len(sample)/100)
N_bins= sturges(sample)
x_range= (xMin,xMax)
bin_content,bin_edges = np.histogram(sample,bins=N_bins,range=x_range)



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


###Function pdf of a exponential###

def exp_pdf(x,tau):
    if tau ==0.: return 1.
    else: return np.exp(-1*x/tau)/tau
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
###Read data from a .txt file###

with open('data.txt') as o:
    sample = [float(x) for x in o.readlines()]
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# Load data from "data.txt" as float values using numpy 
data = np.genfromtxt("data.txt")
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
##Load data using numpy if data.txt are CSV format
data = np.loadtxt("data.txt", delimiter=",")

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# Example of multiple plots

def calculate_mean_sigma(data, M, Gamma):
    iGammas = np.arange(1, 101, dtype=int)
    means = []
    sigmas = []

    for i in iGammas:
        interval_data = data[(data > M - i * Gamma) & (data < M + i * Gamma)]
        means.append(np.mean(interval_data))
        sigmas.append(np.std(interval_data))

    return iGammas, means, sigmas

# Esempio d'uso:
M = 0
Gamma = 1
N = 10000

iGammas, means, sigmas = calculate_mean_sigma(sample, M, Gamma)

# Grafico della media e sigma in funzione di un'altro valore che varia iGamma
def plot_mean_sigma(iGammas, means, sigmas):
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    axs[0].plot(iGammas, means, 'o-', label='Mean')
    axs[0].set_xlabel('$i\Gamma$')
    axs[0].set_ylabel('Mean')
    axs[0].legend()

    axs[1].plot(iGammas, sigmas, 'o-', label='Sigma')
    axs[1].set_xlabel('$i\Gamma$')
    axs[1].set_ylabel('Sigma')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('mean_sigma.png')
    plt.show()

plot_mean_sigma(iGammas, means, sigmas)

    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    




