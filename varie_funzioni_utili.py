# come concatenare vari dati con numpy#

dati_combinati = np.concatenate((dati_exp, dati_gauss))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#genera eventi secondo distribuzione Poissoniana usando numpy#

N_poiss = np.random.poisson(lam = 7, size = 10000)

plt.hist(N_poiss, bins = 30, color = "Purple")
plt.grid()
plt.title("Distribuzione di Poisson")
plt.xlabel("Valori")
plt.ylabel("N° Eventi")
plt.show()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def poissonian(lambda_value=10, N_exp=2000):
    """
    Generate a sample of Poisson-distributed events and plot the histogram using Sturges' rule for bins.

    Parameters:
    lambda_value (float): The average rate of events (λ).
    N_exp (int): The number of events to generate.

    Returns:
    sample (ndarray): The generated sample of Poisson-distributed events.
    bin_edges (ndarray): The edges of the bins used in the histogram.
    """
    # Generate the sample of Poisson-distributed events
    sample = np.random.poisson(lam=lambda_value, size=N_exp)
    
    # Calculate the number of bins using Sturges' rule
    num_bins = int(np.ceil(np.log2(N_exp) + 1))
    
    # Plot the histogram of the sample
    counts, bin_edges, _ = plt.hist(sample, bins=num_bins, density=True, alpha=0.75, color='Purple', edgecolor='black')
    plt.title(f'Poisson Distribution (λ={lambda_value}, N={N_exp})')
    plt.xlabel('Number of events')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()
    
    return sample, bin_edges

# Example usage
sample, bin_edges = poissonian(lambda_value=10, N_exp=2000)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#TOY EXPERIMENT
#generazione di un sample di eventi secondo una distribuzione uniforme, per osservare distr delle means#

from myrand import generate_uniform
# ....
means = []
# loop over toys
for i in range (N_toy):
    randlist = generate_uniform (N_evt)
    toy_stats = stats (randlist)
    means.append (toy_stats.mean ())

#visualizzazione della distribuzione delle media di un sample#

from stats import stats
# ...
means_stats = stats (means)
xMin = means_stats.mean () - 5. * means_stats.sigma ()
xMax = means_stats.mean () + 5. * means_stats.sigma ()
nBins = floor (len (means) / 10.) + 1     # number of bins of the histogram
bin_edges = np.linspace (xMin, xMax, nBins + 1)  # edges o the histogram bins
fig, ax = plt.subplots ()
ax.set_title ('Histogram of the mean over ' + str (N_toy) + ' toys', size=14)
ax.set_xlabel ('mean value')
ax.set_ylabel ('toys in bin')
ax.hist (means,
         bins = bin_edges,
         color = 'orange',
        )
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#compare con la std della media#

print ('sigma of the means disitribution:             ', means_stats.sigma ())
print ('mean of the sigma of the means disitribution: ', sigma_means_stats.mean ())

# plot the distribution of the sigma on the mean
# calculated for each toy
xMin = sigma_means_stats.mean () - 5. * sigma_means_stats.sigma ()
xMax = sigma_means_stats.mean () + 5. * sigma_means_stats.sigma ()
nBins = floor (len (sigma_means) / 10.) + 1     # number of bins of the histogram
bin_edges = np.linspace (xMin, xMax, nBins + 1)  # edges o the histogram bins
fig, ax = plt.subplots ()
ax.set_title ('Histogram of the sigma on the mean over ' + str (N_toy) + ' toys', size=14)
ax.set_xlabel ('mean value')
ax.set_ylabel ('toys in bin')
ax.hist (sigma_means,
         bins = bin_edges,
         color = 'orange',
        )
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


#Esempio di Likelihood 


def exp_pdf (x, tau) :      
    '''
    the exponential probability density function
    '''
    if tau == 0. : return 1.
    if x < 0. : return 0. 
    return exp (-1 * x / tau) / tau

def loglikelihood(theta, pdf, sample):
    logL = 0
    for _ in sample:
        if (pdf(x,theta) > 0): logL += np.log(pdf(x,theta))
    return logL
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----



