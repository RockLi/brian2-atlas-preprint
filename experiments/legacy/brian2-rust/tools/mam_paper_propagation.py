"""Fig. 7 lag convention; temporal order of fluctuations, not causal influence.

MAM commit 0a658be40bef3249cbe452f38809edf7d2f524ba; toolbox arithmetic
checked against 26b9e999069990a8b756d8a4d880bd152f95149f. Historical
installation identity is not established by this source-date match alone.
"""
import numpy as np
from scipy.signal import find_peaks_cwt

SETTINGS=dict(bin_ms=1.,toolbox_commit='26b9e999069990a8b756d8a4d880bd152f95149f',
    mam_commit='0a658be40bef3249cbe452f38809edf7d2f524ba',cross_spectrum='FFT(A) * conjugate(FFT(B)) * 1000 / sample_count',
    boundary='circular FFT covariance; no zero padding',
    time_axis='arange(-n/2+1,n/2+1) / (2*max(fftfreq(n,.001))) * 1000',
    gaussian_sigma_ms=2.,gaussian_samples=list(range(-5,5)),gaussian_convolution='numpy same',
    peak_width_samples=5.,lag_interval_ms='(-100,100)',extremum='largest absolute candidate; minimum on a tie',
    reciprocal_rule='reuse negated reciprocal if nonzero; recompute if zero',excluded_area='MDP',causal_inference=False)


def transforms(rates):
    x=np.asarray(rates,dtype=np.float64)
    if (x.ndim!=2 or not 1<=len(x)<=32 or x.shape[1] not in (10000,100000)
            or not np.isfinite(x).all()):
        raise ValueError('finite full 10 or 100 s area-rate arrays required')
    # Match per-area 1D centralize(..., units=True).
    return np.fft.fft(x-x.mean(axis=1,keepdims=True),axis=1)


def time_axis(n):
    if n not in (10000,100000):raise ValueError('10 or 100 s sample count required')
    freq=np.fft.fftfreq(n,.001)
    step=1./(2.*np.max(freq))*1e3
    return np.arange(-n/2.+1,n/2.+1)*step


def smoothed_curve(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if (a.ndim!=1 or a.shape!=b.shape or len(a) not in (10000,100000)
            or a.dtype.kind!='c' or b.dtype.kind!='c' or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('bounded matching complex spectra required')
    cross=(a*b.conj())*(1./len(a)*1e3)
    raw=np.real(np.fft.ifft(cross));mid=len(raw)//2
    shifted=np.hstack([raw[mid+1:],raw[:mid+1]])
    t=np.arange(-5.,5.);sigma=2.
    kernel=1/(np.sqrt(2.*np.pi)*sigma)*np.exp(-(t**2/(2*sigma**2)))
    return np.convolve(kernel,shifted,mode='same')


def select_peak(time,curve,*,same_area=False,excluded=False):
    time,curve=np.asarray(time),np.asarray(curve)
    if time.ndim!=1 or time.shape!=curve.shape or not np.isfinite(time).all() or not np.isfinite(curve).all():
        raise ValueError('finite matching lag curve required')
    if excluded:return np.nan,np.nan,'excluded_MDP'
    indices=np.flatnonzero((time>-100.)&(time<100.));t=time[indices];cc=curve[indices]
    if len(t)<20 or len(np.flatnonzero(t==0))!=1:raise ValueError('complete central lag window with zero required')
    candidates=[]
    for sign in [1.,-1.]:
        signal=sign*cc;peaks=find_peaks_cwt(signal,np.array([5.]))
        if len(peaks):
            index=int(np.flatnonzero(t==0)[0]) if same_area else int(peaks[np.argmax(signal[peaks])])
            candidates.append((index,float(signal[index])))
        else:candidates.append((None,0.))
    chosen=candidates[0] if abs(candidates[0][1])>abs(candidates[1][1]) else candidates[1]
    if chosen[0] is None:return np.nan,np.nan,'no_selected_extremum'
    index=chosen[0]
    return float(t[index]),float(cc[index]),'selected'


def lag_matrix(rates,areas):
    fft=transforms(rates);count=len(fft)
    if len(areas)!=count or len(set(areas))!=count:raise ValueError('unique area order required')
    time=time_axis(fft.shape[1]);lags=np.zeros((count,count));status={};computed=0
    for i,area in enumerate(areas):
        for j,other in enumerate(areas):
            if lags[j,i]!=0.:
                lags[i,j]=-lags[j,i]
            elif area=='MDP' or other=='MDP':
                lags[i,j]=np.nan;status[area+'/'+other]='excluded_MDP'
            else:
                curve=smoothed_curve(fft[i],fft[j]);computed+=1
                lag,peak,state=select_peak(time,curve,same_area=(i==j))
                lags[i,j]=lag
                if state!='selected':status[area+'/'+other]=state
    return lags,dict(computed_curves=computed,unavailable=status,lag_step_ms=float(time[1]-time[0]))


def fit_hierarchy(lags):
    """Exact minimizer of the figure's complete-pair least-squares objective.

    Fix the otherwise arbitrary additive offset to zero mean. Do not silently
    omit unknown pairs or force an antisymmetric matrix: the figure's zero-lag
    reciprocal rule can yield asymmetric pairs after smoothing/peak detection.
    """
    c=np.asarray(lags,dtype=np.float64)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not 2<=len(c)<=32 or not np.isfinite(c).all():
        raise ValueError('complete finite lag matrix required after declared area exclusion')
    h=(c.sum(axis=1)-c.sum(axis=0))/(2.*len(c))
    residual=h[:,None]-h[None,:]-c
    spread=float(np.ptp(h))
    return dict(levels_ms=h,normalized_levels=(h-h.min())/spread if spread>0 else np.full(len(h),np.nan),
                residual_norm=float(np.linalg.norm(residual)),rmse_ms=float(np.sqrt(np.mean(residual**2))),
                normalization_defined=(spread>0))
