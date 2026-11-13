"""TS4-only adapter of the supplied Welch/Huber/multiple-coherence method.

The reference's actual functions are loaded by AST without running imports,
loaders, plotting or its multi-rate example. Point estimates retain the original
Hann Fourier coefficients and elementwise complex spectrum averaging. Long
records use deterministic weighted delete-group jackknife diagnostics, retaining
every spectral segment in the full-data point estimate. This is explicitly not
the reference's quadratic-cost delete-one uncertainty calculation on long data.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.signal import get_window

from impedance import impedance_products, save_estimate, load_estimate

MU0 = 4*np.pi*1e-7
REFERENCE_NAME = '0_Com_App_phase_Way1-3-Muti_Coh.py'
FUNCTIONS = ('robust_complex_mean', '_Z_from_segments', 'multiple_coherence',
             '_err_jackknife', '_err_bootstrap', '_err_analytic',
             'estimate_impedance', 'calc_rhoa_phase')
DEFAULTS = {
    'n_cycles':8, 'min_nperseg':512, 'overlap':0.5,
    'robust':'huber', 'coherence_threshold':None,
    'min_windows':4, 'max_condition_normalized_hh':1e10,
    'hermitian_tolerance':1e-10, 'psd_tolerance':1e-10,
    'error_method':'jackknife', 'max_jackknife_groups':20,
    'n_boot':200, 'fft_batch_size':64, 'reference_path':None,
}


def validate_settings(settings=None):
    unknown=set(settings or {})-set(DEFAULTS)
    if unknown:raise ValueError('Unknown Welch settings: '+str(sorted(unknown)))
    s=dict(DEFAULTS);s.update(settings or {})
    for key in ('n_cycles','min_nperseg','min_windows','max_jackknife_groups','n_boot','fft_batch_size'):
        value=s[key]
        if isinstance(value,bool) or int(value)!=value or value<1:
            raise ValueError(key+' must be a positive integer')
        s[key]=int(value)
    if s['min_nperseg']<32 or s['min_windows']<4 or s['max_jackknife_groups']<4:
        raise ValueError('Require min_nperseg>=32, min_windows>=4, jackknife groups>=4')
    if not 0<=s['overlap']<1:raise ValueError('overlap must be [0,1)')
    if s['robust'] not in ('mean','median','huber'):raise ValueError('Unknown robust mean')
    if s['error_method'] not in ('jackknife','bootstrap','analytic',None):raise ValueError('Unknown uncertainty method')
    if s['coherence_threshold'] is not None and not 0<=s['coherence_threshold']<=1:
        raise ValueError('coherence_threshold must be None or [0,1]')
    for key in ('max_condition_normalized_hh','hermitian_tolerance','psd_tolerance'):
        if not np.isfinite(s[key]) or s[key]<=0:raise ValueError('Invalid '+key)
    preferred=Path(__file__).resolve().parents[1]/'reference'/'user_way1_3_multi_coh.py'
    s['reference_path']=str(Path(s['reference_path'] or preferred).resolve())
    return s


@lru_cache(maxsize=4)
def reference_functions(path):
    """Compile only named function definitions; the reference main never runs."""
    path=Path(path)
    source=path.read_text(encoding='utf-8-sig')
    tree=ast.parse(source,filename=str(path))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in FUNCTIONS]
    if {n.name for n in selected}!=set(FUNCTIONS):raise ValueError('Reference functions are incomplete')
    namespace={'np':np,'get_window':get_window,'MU0':MU0}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    namespace['_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return namespace


def default_frequencies(n_time, fs=150.0, long_record=False, settings=None):
    """Keep the same requested TS4 grid as the existing Way 2 supplement.

    Unlike Way 2's shorter DPSS windows, some requested short-clip frequencies
    do not have four Welch windows. They remain in the output as rejected rows.
    """
    if not np.isfinite(fs) or abs(float(fs)-150)>1e-12:raise ValueError('Only TS4 fs=150 Hz is supported')
    reference=np.logspace(np.log10(1000.),np.log10(1/2040.8163265306123),66)[13:31]
    if long_record:reference=np.r_[reference,np.geomspace(reference[-1],.1,13)[1:]]
    return reference


def window_geometry(n_time, fs, requested_frequency, settings=None):
    s=validate_settings(settings)
    if not np.isfinite(requested_frequency) or not 0<requested_frequency<fs/2:
        return {'nperseg':0,'step':0,'nfft':0,'bin_index':-1,'frequency_actual_hz':np.nan,'n_segments':0}
    length=min(max(int(round(fs*s['n_cycles']/requested_frequency)),s['min_nperseg']),int(n_time))
    if length<1:return {'nperseg':length,'step':0,'nfft':0,'bin_index':-1,'frequency_actual_hz':np.nan,'n_segments':0}
    step=max(int(length*(1-s['overlap'])),1)
    nfft=int(2**np.ceil(np.log2(length)))
    bins=np.fft.rfftfreq(nfft,d=1/fs)
    k=int(np.argmin(abs(bins-requested_frequency)))
    return {'nperseg':length,'step':step,'nfft':nfft,'bin_index':k,
            'frequency_actual_hz':float(bins[k]),'n_segments':int((n_time-length)//step+1) if n_time>=length else 0}


def fourier_segments(data, geometry, batch_size=64):
    """Same segment demeaning, periodic Hann and rFFT bin as reference loops."""
    length,step,nfft,k=(geometry[v] for v in ('nperseg','step','nfft','bin_index'))
    starts=np.arange(0,data.shape[-1]-length+1,step,dtype=np.int64)
    window=get_window('hann',length)
    values=np.empty((len(starts),4),complex)
    offsets=np.arange(length,dtype=np.int64)
    for j in range(0,len(starts),batch_size):
        selected=starts[j:j+batch_size,None]+offsets[None,:]
        seg=np.transpose(data[:,selected],(1,0,2))
        seg=(seg-seg.mean(axis=2,keepdims=True))*window[None,None,:]
        values[j:j+len(selected)]=np.fft.rfft(seg,n=nfft,axis=2)[:,:,k]
    return values[:,:2],values[:,2:]


def matrix_diagnostics(shh, hermitian_tolerance=1e-10, psd_tolerance=1e-10):
    """Diagnose, never project or silently repair, the spectrum being inverted."""
    a=np.asarray(shh,complex)
    if a.shape!=(2,2) or not np.isfinite(a).all():
        return {'valid':False,'reasons':['nonfinite_magnetic_spectrum'],'rank_hh':0,
                'condition_number_normalized_hh':np.inf,'normalized_hh_min_eigenvalue':np.nan,
                'hermitian_relative_error':np.nan}
    norm=float(np.linalg.norm(a))
    asym=float(np.linalg.norm(a-a.conj().T)/max(norm,np.finfo(float).tiny))
    powers=np.diag(a).real
    reasons=[]
    if asym>hermitian_tolerance:reasons.append('nonhermitian_robust_magnetic_spectrum')
    if np.any(powers<=0):
        return {'valid':False,'reasons':reasons+['nonpositive_magnetic_power'],'rank_hh':0,
                'condition_number_normalized_hh':np.inf,'normalized_hh_min_eigenvalue':np.nan,
                'hermitian_relative_error':asym}
    normalized=a/np.sqrt(powers[:,None]*powers[None,:])
    # Hermitian symmetrization is for diagnostics only. The source inverse is
    # computed from the untouched source SHR, never this diagnostic matrix.
    eig=np.linalg.eigvalsh((normalized+normalized.conj().T)/2)
    rank=int(np.linalg.matrix_rank(normalized,tol=1e-12))
    condition=float(np.linalg.cond(normalized))
    if float(eig.min()) < -psd_tolerance:reasons.append('indefinite_robust_magnetic_spectrum')
    if rank<2:reasons.append('rank_deficient_magnetic_spectrum')
    return {'valid':not reasons,'reasons':reasons,'rank_hh':rank,
            'condition_number_hh':float(np.linalg.cond(a)),
            'condition_number_normalized_hh':condition,'normalized_hh_min_eigenvalue':float(eig.min()),
            'normalized_hh_max_eigenvalue':float(eig.max()),'hermitian_relative_error':asym}


def weighted_group_jackknife(ser,shh,z_full,reference,robust='huber',max_groups=20,quality_settings=None):
    """Exact source delete-one for small n; all-segment weighted block jackknife.

    For n>max_groups, contiguous nearly equal blocks partition EVERY segment.
    With a_b=m_b/n and pseudo p_b=[theta-(1-a_b)*theta_-b]/a_b,
    theta_J=sum(a_b*p_b), var=(1/B)*sum[a_b/(1-a_b)*|p_b-theta_J|^2].
    Equal blocks reduce to ordinary delete-group jackknife. The diagnostic
    uncertainty does not turn overlapping windows into independent observations.
    """
    n=len(ser);s=validate_settings(quality_settings)
    exact=n<=max_groups
    groups=[np.asarray([i]) for i in range(n)] if exact else np.array_split(np.arange(n),max_groups)
    fractions=np.array([len(g)/n for g in groups])
    pseudo=np.empty((len(groups),2,2),complex)
    deleted_values=np.empty_like(pseudo);invalid=[]
    for i,(group,a) in enumerate(zip(groups,fractions)):
        keep=np.ones(n,bool);keep[group]=False
        try:
            deleted,_,hh=reference['_Z_from_segments'](ser[keep],shh[keep],robust=robust)
            diag=matrix_diagnostics(hh,s['hermitian_tolerance'],s['psd_tolerance'])
            reasons=list(diag['reasons'])
            if diag['condition_number_normalized_hh']>s['max_condition_normalized_hh']:
                reasons.append('ill_conditioned_magnetic_spectrum')
            if not np.isfinite(deleted).all():reasons.append('nonfinite_impedance')
            if reasons:
                invalid.append({'group':i,'size':len(group),'first_segment':int(group[0]),
                                'last_segment':int(group[-1]),'reasons':reasons})
                continue
        except (ValueError,RuntimeWarning,np.linalg.LinAlgError,FloatingPointError) as error:
            invalid.append({'group':i,'size':len(group),'reasons':['replica_estimation_failure: '+str(error)]})
            continue
        deleted_values[i]=deleted
        pseudo[i]=(z_full-(1-a)*deleted)/a
    info={'method':'exact_reference_delete_one_jackknife' if exact else 'weighted_contiguous_delete_group_jackknife',
        'groups':len(groups),'group_sizes':[len(g) for g in groups],'group_fractions':fractions.tolist(),
        'all_segments_retained':True,'replica_quality_checks':'same Hermitian/PSD/rank/condition rules as full-data spectrum',
        'valid_replicates':len(groups)-len(invalid),'invalid_replicates':invalid}
    if invalid:
        info['warning']='Uncertainty undefined because at least one delete-group replica failed; full-data point estimate remains accepted'
        return np.full((2,2),np.nan),info
    if exact:
        # This is exactly the source _err_jackknife arithmetic, using the
        # already computed, checked leave-one-out point estimates.
        center=deleted_values.mean(axis=0)
        variance=(n-1)/n*np.sum(abs(deleted_values-center)**2,axis=0)
        return np.sqrt(variance),info
    center=np.sum(fractions[:,None,None]*pseudo,axis=0)
    variance=np.sum((fractions/(1-fractions))[:,None,None]*abs(pseudo-center)**2,axis=0)/len(groups)
    info.update(formula='p_b=(theta-(1-a_b)*theta_minus_b)/a_b; thetaJ=sum(a_b*p_b); var=sum(a_b/(1-a_b)*abs(p_b-thetaJ)^2)/B',
                reference='https://reich.hms.harvard.edu/sites/reich.hms.harvard.edu/files/inline-files/wjack.pdf')
    return np.sqrt(variance),info


def estimate_impedance(data, fs=150.0, frequencies=None, settings=None, unit='si', long_record=False):
    tic=time.perf_counter();s=validate_settings(settings)
    if not np.isfinite(fs) or abs(float(fs)-150)>1e-12:raise ValueError('Only TS4 sampling rate 150 Hz is supported')
    if unit not in ('si','field'):raise ValueError('unit must be si or field')
    x=np.array(data,dtype=np.float64,copy=True)
    if x.ndim!=2 or x.shape[0]!=4 or x.shape[1]<16 or not np.isfinite(x).all():
        raise ValueError('Expected finite [4,time] Ex,Ey,Hx,Hy')
    x-=x.mean(axis=1,keepdims=True)
    f=default_frequencies(x.shape[1],fs,long_record,s) if frequencies is None else np.asarray(frequencies,float)
    if f.ndim!=1 or not len(f) or len(np.unique(f))!=len(f):raise ValueError('Need unique requested frequency vector')
    source=reference_functions(s['reference_path'])
    geometry=[window_geometry(x.shape[1],fs,float(v),s) for v in f]
    actual=np.asarray([g['frequency_actual_hz'] for g in geometry])
    finite=actual[np.isfinite(actual)]
    if len(np.unique(finite))!=len(finite):
        raise ValueError('Requested frequencies collapse to duplicate FFT bins; choose a unique realized grid')
    z=np.full((len(f),2,2),np.nan+1j*np.nan,complex)
    z_unfiltered=np.full_like(z,np.nan+1j*np.nan)
    sigma=np.full(z.shape,np.nan)
    accepted=np.zeros(len(f),bool)
    n_segments=np.asarray([g['n_segments'] for g in geometry],int);n_used=np.zeros(len(f),int)
    coh_ex=np.full(len(f),np.nan);coh_ey=coh_ex.copy();coh_mean=coh_ex.copy()
    qc=[];rng=np.random.default_rng(12345)
    for i,(requested,g) in enumerate(zip(f,geometry)):
        row={'requested_frequency_hz':float(requested),'frequency_actual_hz':float(actual[i]),
             'frequency_offset_hz':float(actual[i]-requested),**g,'accepted':False,'reasons':[]}
        if not np.isfinite(requested) or not 0<requested<fs/2:
            row['reasons'].append('outside_open_Nyquist_interval');qc.append(row);continue
        if not np.isfinite(actual[i]) or not 0<actual[i]<fs/2:
            row['reasons'].append('actual_FFT_bin_outside_open_Nyquist_interval');qc.append(row);continue
        if g['nperseg']<32 or g['n_segments']<s['min_windows']:
            row['reasons'].append('insufficient_Welch_segments');qc.append(row);continue
        e,h=fourier_segments(x,g,batch_size=s['fft_batch_size'])
        n_segments[i]=len(e)
        cx,cy=source['multiple_coherence'](e,h)
        coh_ex[i]=cx;coh_ey[i]=cy;coh_mean[i]=float(np.nanmean([cx,cy]))
        row.update(coherence_ex=float(cx),coherence_ey=float(cy),coherence_mean=float(coh_mean[i]),
                   coherence_semantics='frequency-level ordinary-mean multiple coherence, diagnostic unless threshold configured')
        if s['coherence_threshold'] is not None and np.isfinite(coh_mean[i]) and coh_mean[i]<s['coherence_threshold']:
            row['reasons'].append('frequency_multiple_coherence_below_threshold');qc.append(row);continue
        ser=np.einsum('na,nb->nab',e,np.conj(h))
        shh=np.einsum('na,nb->nab',h,np.conj(h))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',RuntimeWarning)
                unfiltered,_,robust_hh=source['_Z_from_segments'](ser,shh,robust=s['robust'])
            z_unfiltered[i]=unfiltered
            diag=matrix_diagnostics(robust_hh,s['hermitian_tolerance'],s['psd_tolerance'])
            row.update({k:v for k,v in diag.items() if k!='reasons'})
            row['reasons'].extend(diag['reasons'])
            if diag['condition_number_normalized_hh']>s['max_condition_normalized_hh']:
                row['reasons'].append('ill_conditioned_magnetic_spectrum')
            if not np.isfinite(unfiltered).all():row['reasons'].append('nonfinite_impedance')
            if row['reasons']:qc.append(row);continue
            # A resampling failure must not erase a valid full-data tensor.
            z[i]=unfiltered;accepted[i]=True;n_used[i]=len(e)
            row.update(accepted=True,used_segments=len(e))
            error_info={'method':'none'}
            try:
                if s['error_method']=='jackknife':
                    sig,error_info=weighted_group_jackknife(ser,shh,unfiltered,source,s['robust'],s['max_jackknife_groups'],s)
                elif s['error_method']=='bootstrap':
                    sig=source['_err_bootstrap'](ser,shh,robust=s['robust'],n_boot=s['n_boot'],rng=rng)
                    error_info={'method':'reference_bootstrap_std_abs_Z','n_boot':s['n_boot'],
                                'limitation':'Magnitude scatter only; not full complex covariance'}
                elif s['error_method']=='analytic':
                    sig=source['_err_analytic'](ser,unfiltered)
                    error_info={'method':'reference_SER_dispersion_approximation','limitation':'Not a strict Sims-Bostick/Gamble formula'}
                else:sig=np.full((2,2),np.nan)
            except (ValueError,RuntimeWarning,np.linalg.LinAlgError,FloatingPointError) as error:
                sig=np.full((2,2),np.nan)
                error_info={'method':s['error_method'],'failure':str(error),
                            'warning':'Uncertainty failed; full-data point estimate remains accepted'}
            sigma[i]=sig;row.update(uncertainty=error_info)
            if s['error_method'] is not None and not np.isfinite(sig).all():
                row['uncertainty_warning']='Some uncertainty components are undefined; point estimate retained'
        except (ValueError,RuntimeWarning,np.linalg.LinAlgError,FloatingPointError) as error:
            row['reasons'].append('estimation_failure: '+str(error))
        qc.append(row)
    safe_actual=np.where(np.isfinite(actual)&(actual>0)&(actual<fs/2),actual,np.nan)
    products=impedance_products(z,sigma,safe_actual,unit=unit)
    metadata={'method':'Welch / Hann / elementwise %s cross-power-spectrum impedance' % s['robust'],
        'reference_script':REFERENCE_NAME,'reference_source_sha256':source['_source_sha256'],
        'source_function_loading':'AST-selected untouched functions; no example I/O executed',
        'sampling_rate_hz':float(fs),'source_samples':int(x.shape[-1]),
        'channels':['Ex','Ey','Hx','Hy'],'tensor_convention':'E = Z H; rows Ex/Ey, columns Hx/Hy',
        'settings':s,'remote_reference':False,'full_data_point_estimate':'All available Welch segments; no downsampling',
        'coherence':'Frequency-level ordinary-mean Ex/Ey multiple coherence; default threshold None',
        'frequency_contract':'frequency holds requested IDs; frequency_actual_hz/f_used_hz is the realized FFT bin and is used in rho/plots',
        'unit_convention':unit,'rho_frequency':'realized FFT frequency, not nominal target',
        'rho_formula':'abs(Z)^2/(2*pi*f_actual*mu0)' if unit=='si' else '0.2*abs(Z)^2/f_actual',
        'calibration_assumption':'User requested already-calibrated amplitudes; no extra x1000 or per-channel standardization',
        'phase':'Raw complex argument stored in phase_deg; reference-folded MT display also stored in phase_mt_deg',
        'uncertainty':'Reference exact delete-one for <=max_groups segments; weighted contiguous delete-group for longer records; diagnostic, not independent-window CI',
        'quality':'Reject insufficient, nonfinite, non-Hermitian, indefinite/rank-deficient/ill-conditioned magnetic spectra; no PSD repair',
        'elapsed_seconds':time.perf_counter()-tic}
    return {'frequency':f,'frequencies_hz':f,'frequency_requested_hz':f,
            'frequency_actual_hz':actual,'f_used_hz':actual,
            'z':z,'z_unfiltered':z_unfiltered,'z_sigma':sigma,'sigma_z':sigma,
            'accepted':accepted,'n_segments':n_segments,'n_used':n_used,
            'coherence_ex':coh_ex,'coherence_ey':coh_ey,'coherence_mean':coh_mean,
            'qc':qc,'metadata':metadata,**products}


estimate=estimate_impedance
