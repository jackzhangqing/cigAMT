"""Razorback/DPSS/M-estimator adapter for four-channel TS4 signals.

This adapter keeps the user's Way 2 Fourier and regression implementation.
It changes array I/O, frequency admissibility and explicit quality reporting.
No channel standardization, learned denoising or calibration is performed here.

Default apparent-resistivity conversion matches the active SI expression in
the supplied ``resistivity_phase2.py``. Its applicability to the input units is
an explicit user-confirmed calibration assumption, not a new calibration audit.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
import warnings

import numpy as np

MU0 = 4.0 * np.pi * 1e-7
TAGS = {'Ex': (0,), 'Ey': (1,), 'Hx': (2,), 'Hy': (3,),
        'E': (0, 1), 'B': (2, 3), 'site_1': (0, 1, 2, 3)}
DEFAULTS = {
    'n_periods': 4, 'overlap': 0.71, 'slepian_tau': 2.0,
    'min_windows': 8, 'min_used_windows': 4,
    'max_condition_normalized_hh': 1e10,
    'coherence_threshold': None,
    'mest_tolerance': 1e-2, 'mest_max_iterations': 1000,
    'mest_epsilon': 1e-14,
    'vendor_root': None,
}


def validate_settings(settings=None):
    unknown = set(settings or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown impedance settings: %s' % sorted(unknown))
    s = dict(DEFAULTS)
    s.update(settings or {})
    for k in ('n_periods', 'min_windows', 'min_used_windows', 'mest_max_iterations'):
        if isinstance(s[k], bool) or int(s[k]) != s[k] or s[k] < 1:
            raise ValueError('%s must be a positive integer' % k)
        s[k] = int(s[k])
    if s['min_windows'] < 4 or s['min_used_windows'] < 3:
        raise ValueError('Require min_windows >= 4 and min_used_windows >= 3')
    if not 0 <= s['overlap'] < 1:
        raise ValueError('overlap must be in [0,1)')
    for k in ('slepian_tau', 'mest_tolerance', 'mest_epsilon', 'max_condition_normalized_hh'):
        if not np.isfinite(s[k]) or s[k] <= 0:
            raise ValueError('%s must be finite and positive' % k)
    if s['coherence_threshold'] is not None and not 0 <= s['coherence_threshold'] <= 1:
        raise ValueError('coherence_threshold must be None or in [0,1]')
    s['vendor_root'] = str(Path(s['vendor_root'] or Path(__file__).resolve().parents[1] / 'vendor').resolve())
    return s


@lru_cache(maxsize=4)
def _backend(root):
    root = Path(root).resolve()
    for name in ('razorback', 'razorback_v04'):
        if not (root / name / '__init__.py').is_file():
            raise FileNotFoundError('Missing vendored %s at %s' % (name, root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import razorback as rb
    import razorback_v04 as rv
    from razorback_v04.fourier_transform import slepian_window
    for package in (rb, rv):
        if root not in Path(package.__file__).resolve().parents:
            raise RuntimeError('Unexpected backend import: %s' % package.__file__)
    return rb, rv, slepian_window


def window_geometry(n_time, fs, frequency, n_periods=4, overlap=.71):
    """Actual sliding-view sample count; invalid geometry has zero windows."""
    if not np.isfinite(frequency) or not 0 < frequency < fs / 2:
        return 0, 0, 0
    length = int(math.ceil(n_periods * fs / frequency))
    shift = int(length * (1.0 - overlap))
    if shift < 1 or length > n_time:
        return 0, length, shift
    return int((n_time-length)//shift+1), length, shift


def default_frequencies(n_time, fs=150.0, long_record=False, settings=None):
    """Use reference TS4 medium grid, optionally extending long clips to 0.1 Hz.

    No below-support frequency is silently retained. The reference grid is the
    active 66-point logarithmic grid in the supplied Way 2 script, indices13:31.
    """
    s = validate_settings(settings)
    reference = np.logspace(np.log10(1000.0), np.log10(1/2040.8163265306123), 66)
    frequencies = reference[13:31]
    if long_record:
        frequencies = np.r_[frequencies, np.geomspace(frequencies[-1], .1, 13)[1:]]
    keep = [f for f in frequencies if window_geometry(n_time, fs, f, s['n_periods'], s['overlap'])[0] >= s['min_windows']]
    if not keep:
        raise ValueError('No admissible frequency for clip length %d' % n_time)
    return np.asarray(keep, dtype=float)


def _diagnostics(coeff):
    e = np.asarray(coeff[:2], dtype=np.complex128)
    h = np.asarray(coeff[2:], dtype=np.complex128)
    n = h.shape[1]
    shh = h @ h.conj().T / n
    see = np.mean(abs(e)**2, axis=1)
    seh = e @ h.conj().T / n
    powers = np.diag(shh).real
    if np.any(powers <= 0) or not np.all(np.isfinite(powers)):
        return {'coherence_ex': np.nan, 'coherence_ey': np.nan,
                'condition_number_hh': np.inf, 'condition_number_normalized_hh': np.inf,
                'rank_hh': 0}
    normalized = shh / np.sqrt(powers[:, None] * powers[None, :])
    rank = int(np.linalg.matrix_rank(normalized, tol=1e-12))
    inv = np.linalg.pinv(shh)
    coh = []
    for row, energy in zip(seh, see):
        value = float((row[None] @ inv @ row.conj()[:, None]).real.item())
        coh.append(float(np.clip(value/energy, 0, 1)) if energy > 0 else np.nan)
    return {'coherence_ex': coh[0], 'coherence_ey': coh[1],
            'condition_number_hh': float(np.linalg.cond(shh)),
            'condition_number_normalized_hh': float(np.linalg.cond(normalized)),
            'rank_hh': rank}


class _CachedSignal:
    """Reuse exactly the computed coefficients for diagnostics and regression."""
    def __init__(self, original, frequency, result):
        self._original, self._frequency, self._result = original, frequency, result
    def __getattr__(self, name):
        return getattr(self._original, name)
    def fourier_coefficients(self, frequency, **kwargs):
        if float(frequency) != float(self._frequency):
            raise ValueError('Unexpected frequency in cached signal wrapper')
        return self._result


def impedance_products(z, sigma_z, frequencies, unit='si'):
    """Return rho and phases for every tensor entry, without channel rescaling."""
    if unit not in ('si', 'field'):
        raise ValueError("unit must be 'si' or 'field'; no relative proxy is used")
    z = np.asarray(z, dtype=np.complex128)
    sigma_z = np.abs(np.asarray(sigma_z, dtype=float))
    f = np.asarray(frequencies, dtype=float)
    if z.shape != (len(f), 2, 2) or sigma_z.shape != z.shape:
        raise ValueError('Expected tensor arrays [frequency,2,2]')
    coefficient = np.full(f.shape, np.nan)
    numerator = 1/(MU0*2*np.pi) if unit == 'si' else .2
    np.divide(numerator, f, out=coefficient, where=np.isfinite(f) & (f>0))
    coefficient = coefficient[:, None, None]
    amplitude = abs(z)
    rho = coefficient * amplitude**2
    rho_sigma = 2*coefficient*amplitude*sigma_z
    phase_raw = np.angle(z, deg=True)
    phase_mt = np.mod(phase_raw, 180.0)
    # Conventional MT display reverses the yx sign before folding to [0,180).
    phase_mt[:, 1, 0] = np.mod(np.angle(-z[:, 1, 0], deg=True), 180.0)
    phase_sigma = np.full_like(amplitude, np.nan)
    np.divide(sigma_z, amplitude, out=phase_sigma, where=amplitude>0)
    phase_sigma *= 180/np.pi
    return {'rho':rho, 'rho_sigma':rho_sigma, 'phase_deg':phase_raw,
            'phase_raw_deg':phase_raw, 'phase_mt_deg':phase_mt,
            'phase_sigma_deg':phase_sigma}


def estimate_impedance(data, fs=150.0, frequencies=None, settings=None, unit='si', long_record=False):
    """Estimate E=Z H for Ex,Ey,Hx,Hy; return arrays plus every rejection.

    Regression uncertainties are Razorback diagnostic errors propagated by
    first-order derivatives, not independent-window confidence intervals.
    """
    s = validate_settings(settings)
    x = np.array(data, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[0] != 4 or x.shape[1] < 16:
        raise ValueError('Expected finite [4,time] with >=16 samples')
    if not np.isfinite(x).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Input and sampling rate must be finite, fs>0')
    x -= x.mean(axis=1, keepdims=True)
    f = default_frequencies(x.shape[1], fs, long_record, s) if frequencies is None else np.asarray(frequencies, dtype=float)
    if f.ndim != 1 or not len(f) or len(np.unique(f)) != len(f):
        raise ValueError('frequencies must be a nonempty unique 1D list')
    rb, rv, slepian = _backend(s['vendor_root'])
    sig = rb.SignalSet(TAGS, rb.SyncSignal(data=tuple(x), sampling_rate=float(fs), start=0, calibrations=None))
    options = {'Nper':s['n_periods'], 'overlap':s['overlap'], 'window':slepian(s['slepian_tau'])}
    z = np.full((len(f),2,2), np.nan+1j*np.nan, dtype=np.complex128)
    sigma = np.full(z.shape, np.nan)
    accepted = np.zeros(len(f), dtype=bool)
    qc = []
    for i, frequency in enumerate(f):
        nw, length, shift = window_geometry(x.shape[1], fs, float(frequency), s['n_periods'], s['overlap'])
        row = {'frequency_hz':float(frequency), 'accepted':False, 'reason':'not_evaluated',
               'n_windows_planned':nw, 'window_length':length, 'shift':shift}
        qc.append(row)
        if not 0 < frequency < fs/2:
            row['reason'] = 'frequency_outside_open_nyquist_interval'; continue
        if nw < s['min_windows']:
            row['reason'] = 'insufficient_windows'; continue
        try:
            coeff, geometry = sig.fourier_coefficients(float(frequency), **options)
            coeff = tuple(np.asarray(v) for v in coeff)
            actual = len(coeff[0]); row['n_windows'] = actual
            row.update(_diagnostics(coeff))
            if actual < s['min_windows']:
                row['reason'] = 'insufficient_actual_windows'; continue
            if row['rank_hh'] < 2 or row['condition_number_normalized_hh'] > s['max_condition_normalized_hh']:
                row['reason'] = 'magnetic_design_rank_or_condition_failure'; continue
            coh = np.asarray([row['coherence_ex'], row['coherence_ey']])
            row['coherence_mean'] = float(np.mean(coh))
            if s['coherence_threshold'] is not None and (not np.isfinite(coh).all() or coh.mean() < s['coherence_threshold']):
                row['reason'] = 'coherence_below_threshold'; continue
            cached = _CachedSignal(sig, frequency, (coeff, geometry))
            capture = io.StringIO()
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(capture):
                warnings.simplefilter('always')
                result = rv.utils.impedance(cached, [float(frequency)], weights=rb.weights.mest_weights,
                    fourier_opts=options, mest_opts={'tol':s['mest_tolerance'], 'maxit':s['mest_max_iterations'], 'eps':s['mest_epsilon']}, silent_fail=True)
            zz = np.asarray(result.impedance[0], dtype=np.complex128)
            ss = np.abs(np.asarray(result.error[0], dtype=float))
            invalid = result.invalid_time[0]
            counts = [len(invalid[j]) if invalid is not None and len(invalid)>j else 0 for j in range(2)]
            row.update(used_ex=actual-counts[0], used_ey=actual-counts[1])
            if caught: row['warnings'] = sorted(set(str(w.message) for w in caught))
            console = capture.getvalue().strip()
            if console and 'starting frequency' not in console: row['backend_message'] = console[:1000]
            if zz.shape != (2,2) or not np.isfinite(zz).all() or not np.isfinite(ss).all():
                row['reason'] = 'nonfinite_tensor_or_regression_error'; continue
            if min(row['used_ex'], row['used_ey']) < s['min_used_windows']:
                row['reason'] = 'insufficient_used_windows'; continue
            z[i], sigma[i], accepted[i] = zz, ss, True
            row.update(accepted=True, reason='accepted')
        except Exception as exc:
            row.update(reason='backend_failure', message='%s: %s' % (type(exc).__name__, exc))
    result = {'frequency':f, 'frequencies_hz':f, 'z':z, 'z_sigma':sigma, 'sigma_z':sigma,
              'accepted':accepted, 'qc':qc}
    result.update(impedance_products(z, sigma, f, unit))
    result['metadata'] = {
        'method':'Razorback single-site DPSS/M-estimator, user Way 2 settings',
        'sampling_rate_hz':float(fs), 'n_samples':int(x.shape[1]), 'unit_convention':unit,
        'channel_order':['Ex','Ey','Hx','Hy'], 'demean':True,
        'per_channel_standardization':False, 'additional_calibration_applied':False,
        'calibration_assumption':'User instructed that input channels are already calibrated; no instrument-response verification is claimed.',
        'rho_formula':'abs(Z)^2/(2*pi*f*mu0)' if unit=='si' else '0.2*abs(Z)^2/f',
        'rho_unit':'ohm m', 'impedance_relation':'E=Z H; rows Ex,Ey; columns Hx,Hy',
        'phase_deg':'raw atan2 in [-180,180]; phase_mt_deg is separately folded for MT display',
        'error_interpretation':'Razorback regression diagnostic with first-order propagation; overlapping windows are not independent confidence samples',
        'settings':s, 'backend_versions':{'razorback':getattr(rb,'__version__',None),'razorback_v04':getattr(rv,'__version__',None)},
        'primary_sources':['https://razorback.readthedocs.io/en/latest/',
                           'https://doi.org/10.3389/feart.2020.00296'],
    }
    return result


def safe_json(value):
    if isinstance(value, dict): return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray): return safe_json(value.tolist())
    if isinstance(value, np.generic): return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, complex): return {'real':safe_json(value.real),'imag':safe_json(value.imag)}
    return value


def save_estimate(result, base):
    """Save numeric results as NPZ plus a strict-JSON QC/provenance companion."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k:v for k,v in result.items() if isinstance(v, np.ndarray)}
    arrays['metadata_json'] = np.asarray(json.dumps(safe_json(result['metadata']), ensure_ascii=False, allow_nan=False))
    arrays['qc_json'] = np.asarray(json.dumps(safe_json(result['qc']), ensure_ascii=False, allow_nan=False))
    np.savez_compressed(base.with_suffix('.npz'), **arrays)
    summary = {'accepted_frequencies':int(np.count_nonzero(result['accepted'])),
               'requested_frequencies':len(result['frequency']),
               'frequencies_hz':result['frequency'], 'metadata':result['metadata'], 'qc':result['qc']}
    base.with_suffix('.json').write_text(json.dumps(safe_json(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return str(base.with_suffix('.npz'))


def load_estimate(path):
    """Read an NPZ produced by save_estimate without object deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        result = {k:archive[k] for k in archive.files if k not in ('metadata_json','qc_json')}
        result['metadata'] = json.loads(str(archive['metadata_json']))
        result['qc'] = json.loads(str(archive['qc_json']))
    return result


# Short alias for command-line pipelines using the previous estimator name.
estimate = estimate_impedance
