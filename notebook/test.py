import matplotlib.pyplot as plt
import numpy as np

from modules import init_cosmology, init_sample, Model

def test_cl_rsd():
    ell = np.arange(2, 500)    
    
    # Anna Porredon
    cl_anna = np.loadtxt('C_ells_bin1_1_linear.txt')

    # Mehdi Rezaie
    cosmo = init_cosmology()    
    z, b, dNdz = init_sample('mock', verb=False)


    th = Model(cosmo)
    th.add_tracer(z, b, dNdz)   
    cl_null = th.compute_cl(ell, fnl=0.0, has_rsd=False, has_fnl=False)
    cl_rsd = th.compute_cl(ell, fnl=0.0, has_rsd=True, has_fnl=False)
    
    ## additional runs
    #cl_fnlp = th.compute_cl(ell, fnl=100.0, has_rsd=True, has_fnl=True)
    #cl_fnln = th.compute_cl(ell, fnl=-100.0, has_rsd=True, has_fnl=True)
    #cls_ccl = run_ccl(cosmo, (z, dNdz), (z, b), ell)
    #cls_ccl_rsd = run_ccl(cosmo, (z, dNdz), (z, b), ell, has_rsd=True)
    
    assert (abs(cl_rsd-cl_anna[2:, 1]) < 1.0e-6).all()   
    
    fig, ax = plt.subplots()
    ax.plot(ell, cl_null, 'C0--', alpha=0.5, label='FFTlog')
    ax.plot(ell, cl_rsd, 'C0-', lw=1, alpha=0.5, label='FFTlog+RSD')
    ax.plot(cl_anna[2:, 0], cl_anna[2:, 1], 'r:', label='Anna')
    
    ## additional curves
    #ax.plot(ell, cl_fnlp, 'C0-.', lw=1, label='FFTlog+RSD (fnl=100)')   
    #ax.plot(ell, cl_fnln, 'C1-.', lw=1, label='FFTlog+RSD (fnl=-100)')    
    #ax.plot(ell, a*cls_ccl, 'C1-', alpha=0.8, label='CCL (Limber)')
    #ax.plot(ell, a*cls_ccl_rsd,'C1--', lw=1, label='CCL+RSD')

    ax.legend(frameon=False, ncol=2, loc='lower left', fontsize=10)
    ax.set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel=r'C$_{\ell}$')
    ax.tick_params(direction='in', which='both', axis='both', right=True, top=True)
    ax.grid(True, ls=':', color='grey', which='both', lw=0.2)

    # ax0 = fig.add_axes([0.2, 0.2, 0.4, 0.3])
    # add_plot(ax0)
    # ax0.set(xlim=(1.9, 5), ylim=(0.9e-6, 2.1e-6))
    # ax0.set_xticks([2, 3, 4])
    ax.set_ylim(8.0e-8, 2.0e-5)
    fig.savefig('cl_fftlog_ccl_benchmark.png', 
                dpi=300, bbox_inches='tight', facecolor='w')    

    
    
def test_pk():
    
    # import pyccl as ccl
    # k = np.logspace(-4, 1, num=1000)
    # pk = ccl.linear_matter_power(cosmo, k, 1.0)
    # np.savetxt('linear_matter_power.txt', np.column_stack([k, pk]), header='k power')
    # np.savetxt('sample_dNdz_bias.txt', np.column_stack([z, dNdz, b]), header='redshift, dNdz, bias')
    
    pk_anna = np.loadtxt('Pk_z0_linear.txt')
    pk_mr = np.loadtxt('linear_matter_power.txt')
    
    # test MR with AP
    _h = 0.6777
    plt.loglog(pk_anna[:, 0], pk_anna[:, 1], 'b-',
               pk_mr[:, 0]/_h, pk_mr[:, 1]*_h**3, 'r--')
    plt.legend(['Anna', 'Mehdi'])    
    