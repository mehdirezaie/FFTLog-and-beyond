
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
from IPython.display import display, Math
import zeus
from modules import (init_cosmology, init_sample, PNGModel,
                     Posterior, read_mocks, WindowSHT, read_weight_mask)

# --- theory
nside = 256
lmax = 3*nside
kind = 'qsomock'
case = 'masked'

cosmo = init_cosmology()
z, b, dNdz = init_sample(kind)
model = PNGModel(cosmo, has_fnl=True, has_rsd=True)
model.add_tracer(z, b, dNdz, p=1.6)

# --- 'Data'
x, y, invcov, cov = read_mocks(case, return_cov=True)
weight, mask = read_weight_mask()

wd = WindowSHT(weight, mask, x, ngauss=lmax)
ell_ = np.arange(lmax)

def modelw(ell, fnl):
    cl_null = model(ell_, fnl)
    cl_null[0] = 0.0
    return wd.convolve(ell_, cl_null)

lg = Posterior(modelw)

for fNL in [-999, -500, -100, -10, 0, 10, 100, 500, 999]:
    print(fNL, lg.logpost(fNL, y, invcov, x))
    
    
np.random.seed(42)

ndim = 1       # Number of parameters/dimensions (e.g. m and c)
nwalkers = 10  # Number of walkers to use. It should be at least twice the number of dimensions.
nsteps = 1000  # Number of steps/iterations.

start = 10. * np.random.randn(nwalkers, ndim) # Initial positions of the walkers.
print(f'initial guess: {start}')

sampler = zeus.EnsembleSampler(nwalkers, ndim, lg.logpost, 
                               args=[y, invcov, x], 
                               maxiter=10000)
sampler.run_mcmc(start, nsteps) # Run sampling
sampler.summary # Print summary diagnostics


# flatten the chains, thin them by a factor of 15, 
# and remove the burn-in (first half of the chain)
chain = sampler.get_chain(flat=True, discard=20, thin=5)

np.save(f'chains_{kind}_{case}.npy', chain)  # 

