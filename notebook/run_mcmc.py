import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import zeus
from modules import init_cosmology, init_sample, Model
import sys
sys.path.append('/Users/rezaie/github/LSSutils')
from lssutils.utils import histogram_cell


# --- theory
cosmo = init_cosmology()
z, b, dNdz = init_sample()
model = Model(cosmo)
model.add_tracer(z, b, dNdz)

# data
cl_mocks = np.load('cl_mocks_1k.npz', allow_pickle=True)
cl_full = cl_mocks['full'] # select full sky mocks
# bin measurements
bins = np.arange(1, 501, 20)

cl_fullb = []
for i in range(cl_full.shape[0]):
    x, clb_ =  histogram_cell(cl_full[i, :], bins=bins)
    cl_fullb.append(clb_)
    #print('.', end='')
cl_fullb = np.array(cl_fullb)
y = cl_fullb.mean(axis=0)

nmocks, nbins = cl_fullb.shape
hf = (nmocks - 1.0)/(nmocks - nbins - 2.0)
cov = np.cov(cl_fullb, rowvar=False)*hf
invcov = np.linalg.inv(cov)
x = x.astype('int')
print(x)

def logprior(theta):
    ''' The natural logarithm of the prior probability. '''
    lp = 0.

    # unpack the model parameters from the tuple
    fnl = theta

    # uniform prior on fNL
    fmin = -200. # lower range of prior
    fmax = 200.  # upper range of prior

    # set prior to 1 (log prior to 0) if in the range and zero (-inf) outside the range
    lp = 0. if fmin < fnl < fmax else -np.inf

    ## Gaussian prior on ?
    #mmu = 3.     # mean of the Gaussian prior
    #msigma = 10. # standard deviation of the Gaussian prior
    #lp -= 0.5*((m - mmu)/msigma)**2

    return lp

def loglike(theta, y, invcov, x):
    '''The natural logarithm of the likelihood.'''
    # unpack the model parameters
    fnl = theta
    # evaluate the model
    md = model(x, fnl=fnl, has_fnl=True, has_rsd=True)
    # return the log likelihood
    return -0.5 * (y-md).dot(invcov.dot(y-md))

def logpost(theta, y, invcov, x):
    '''The natural logarithm of the posterior.'''
    return logprior(theta) + loglike(theta, y, invcov, x)


ndim = 1      # Number of parameters/dimensions (e.g. m and c)
nwalkers = 10 # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 1000 # Number of steps/iterations.
start = 0.01 * np.random.randn(nwalkers, ndim) # Initial positions of the walkers.
print(f'initial guess: {start}')

sampler = zeus.EnsembleSampler(nwalkers, ndim, logpost, args=[y, invcov, x]) # Initialise the sampler
sampler.run_mcmc(start, nsteps) # Run sampling
#sampler.summary # Print summary diagnostics

# flatten the chains, thin them by a factor of 10, and remove the burn-in (first half of the chain)
chain = sampler.get_chain(flat=True, discard=100, thin=10)

np.save('chains.npy', chain)
