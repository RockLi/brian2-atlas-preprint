"""Explicit modern MAM Welch convention; historical environment remains open."""
import inspect
import numpy as np
import scipy
from scipy.signal import welch

SETTINGS = dict(fs=1000.,window='hann_periodic',nperseg=1024,noverlap=1000,
                nfft=1024,detrend='constant',return_onesided=True,scaling='density',average='mean')


def spectrum(rate):
    rate=np.asarray(rate,dtype=np.float64)
    if rate.ndim!=1 or not 1024<=len(rate)<=100000 or not np.isfinite(rate).all():
        raise ValueError('finite 1D rate series of 1024..100000 milliseconds required')
    if scipy.__version__!='1.18.1' or inspect.signature(welch).parameters['window'].default!='hann_periodic':
        raise ValueError('unverified SciPy Welch environment')
    centered=rate-np.mean(rate,axis=0)
    return welch(centered,**SETTINGS)
