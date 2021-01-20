"""
    Second test of FFTLog algorithm


    FFTLog can be used to do an integral similar to:
        
        $$ F_\ell(k) = \int \frac{dr}{r} f(r) j_{\ell}(kr)$$

    if we use $f(r) = r$ for ell=0, then we expect:

        $$ \int dr j_{0}(kr) = \frac{\pi}{2k}$$

"""

import numpy as np
import matplotlib.pyplot as plt
from fftlog import fftlog

r = np.logspace(-4., 2., num=2000)
fr = r

nu = 1.01
ell = 0

myfftlog = fftlog(r, fr, nu=nu, N_extrap_low=1500, N_extrap_high=1500, c_window_width=0.25, N_pad=5000)

k, Fk = myfftlog.fftlog(ell)
Fk_true = np.pi / (2*k)

assert np.all(abs(Fk-Fk_true) < 1.0e-8)

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

ax[0].plot(r, fr)
ax[1].plot(k, Fk, label='FFTLog')
ax[1].plot(k, Fk_true, 'r--', label=r'Truth (i.e., $\pi/(2k)$)')

ax[0].set(xlabel='r', ylabel='f(r)')
ax[1].set(xscale='log', xlabel='k', ylabel='F(k)')
ax[1].legend()
fig.savefig('fftlog_test.png', bbox_inches='tight', dpi=300)
