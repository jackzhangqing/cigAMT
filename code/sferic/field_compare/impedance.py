"""Frequency-local robust transfer estimates without joining separated selections.

The scipy estimator is self-contained; the optional native backend follows the
user's Way2 Razorback workflow. Unit conventions are explicit assumptions, and
do not require calibration files. Per-window z-scores must never be supplied as
physical amplitudes.
"""

from pathlib import Path
import contextlib
import importlib
import io
import re
import sys

import numpy as np
from scipy.signal.windows import dpss


CHANNELS = ("Ex", "Ey", "Hx", "Hy")
MU0 = 4 * np.pi * 1e-7


def merge_intervals(intervals, n_samples):
    """Validate sample [start, stop) intervals, merging strict overlaps only.

    Explicit adjacent intervals can be separate acquisition bursts. Preserve the
    caller's boundary, so neither backend places a Fourier window across it.
    """
    if intervals is None:
        return [(0, int(n_samples))] if n_samples else []
    result = []
    for item in intervals:
        if len(item) != 2:
            raise ValueError("Each interval must contain start and stop")
        start, stop = item
        if not (np.isfinite(start) and np.isfinite(stop)):
            raise ValueError("Interval bounds must be finite integer samples")
        if int(start) != start or int(stop) != stop:
            raise ValueError("Interval bounds must be integer samples")
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= n_samples:
            raise ValueError("Intervals must lie within the source sample range")
        if start != stop:
            result.append((start, stop))
    merged = []
    for start, stop in sorted(result):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def _complex_values(value):
    if isinstance(value, dict) and "real" in value:
        return np.asarray(value["real"], dtype=float) + 1j * np.asarray(value.get("imag", 0), dtype=float)
    return np.asarray(value, dtype=complex)


def _calibration_factors(calibration, frequencies):
    factors = np.ones((len(frequencies), 4), dtype=complex)
    if calibration is None:
        return factors, False
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a mapping or None")
    for key in ("electric_to_mV_per_km", "magnetic_to_nT"):
        value = calibration.get(key)
        if value is None or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError("Physical calibration requires a positive finite " + key)
    factors[:, :2] *= float(calibration["electric_to_mV_per_km"])
    factors[:, 2:] *= float(calibration["magnetic_to_nT"])
    responses = calibration.get("channel_factors")
    if responses is not None:
        response_frequencies = calibration.get("frequency_hz", calibration.get("frequencies"))
        response_count = len(response_frequencies) if response_frequencies is not None else len(frequencies)
        if isinstance(responses, dict) and "real" not in responses:
            columns = []
            for channel in CHANNELS:
                value = _complex_values(responses.get(channel, 1))
                columns.append(np.full(response_count, value.item(), dtype=complex)
                               if value.ndim == 0 else value)
            try:
                response = np.column_stack(columns)
            except ValueError as exc:
                raise ValueError("Channel factor arrays must share the same frequency grid") from exc
        else:
            response = _complex_values(responses)
            if response.shape == (4,):
                response = np.broadcast_to(response, (response_count, 4))
        if response.ndim != 2 or response.shape[1] != 4:
            raise ValueError("channel_factors must have shape (4,) or (n_frequencies,4)")
        if response_frequencies is not None:
            grid = np.asarray(response_frequencies, dtype=float)
            if grid.ndim != 1 or len(grid) != len(response) or not np.isfinite(grid).all():
                raise ValueError("Calibration frequency_hz must match channel_factors")
            order = np.argsort(grid)
            grid, response = grid[order], response[order]
            if len(grid) < 2 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
                raise ValueError("Calibration frequencies must be distinct positive values")
            if np.any(frequencies < grid[0]) or np.any(frequencies > grid[-1]):
                raise ValueError("Requested frequencies lie outside the calibration grid")
            response = np.column_stack([
                np.interp(frequencies, grid, response[:, c].real)
                + 1j * np.interp(frequencies, grid, response[:, c].imag)
                for c in range(4)])
        if response.shape != factors.shape:
            raise ValueError("channel_factors must match the requested frequency count")
        factors *= response
    if not np.isfinite(factors).all() or np.any(np.abs(factors) == 0):
        raise ValueError("Calibration factors must be finite and nonzero")
    return factors, True


def _spectra(sx, intervals, length, hop, frequency, fs_hz):
    """Return exact-frequency tapered Fourier coefficients and interval groups."""
    taper = dpss(length, NW=2, Kmax=1, sym=False)[0]
    kernel = taper * np.exp(-2j * np.pi * frequency * np.arange(length) / fs_hz)
    # Demean each channel before tapering, without copying all sample windows.
    kernel -= kernel.sum() / length
    chunks, groups = [], []
    group = 0
    for start, stop in intervals:
        group += 1
        last_finite = False
        if stop - start < length:
            continue
        view = np.lib.stride_tricks.sliding_window_view(sx[:4, start:stop], length, axis=1)
        count = (stop - start - length) // hop + 1
        batch = max(1, min(4096, 1_000_000 // length))
        for first in range(0, count, batch):
            positions = np.arange(first, min(first + batch, count)) * hop
            block = np.einsum("cwn,n->wc", view[:, positions, :], kernel, optimize=True)
            finite = np.isfinite(block).all(axis=1)
            changes = np.flatnonzero(np.diff(np.r_[False, finite, False]))
            for a, b in zip(changes[::2], changes[1::2]):
                # A skipped nonfinite window is a break for covariance correlations.
                if a or not last_finite:
                    group += 1
                chunks.append(block[a:b])
                groups.append(np.full(b - a, group, dtype=int))
            last_finite = bool(finite[-1])
    if not chunks:
        return np.empty((0, 4), complex), np.empty(0, int)
    return np.concatenate(chunks), np.concatenate(groups)


def _fit_complex(h, e, groups, overlap_lags):
    """Huber IRLS plus interval-local HAC covariance of real/imag coefficients."""
    weights = np.ones(len(e))
    beta = np.zeros(2, complex)
    for _ in range(40):
        root = np.sqrt(weights)
        design = h * root[:, None]
        new_beta, _, rank, singular = np.linalg.lstsq(design, e * root, rcond=None)
        if rank < 2 or singular[-1] <= singular[0] * 1e-10:
            raise np.linalg.LinAlgError("Magnetic spectral design is rank deficient")
        residual = e - h @ new_beta
        scale = max(float(np.median(np.abs(residual))) / np.sqrt(np.log(2)),
                    np.finfo(float).eps * max(1., float(np.median(np.abs(e)))))
        updated = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), np.finfo(float).tiny))
        converged = np.linalg.norm(new_beta - beta) <= 1e-9 * max(1., np.linalg.norm(new_beta))
        beta, weights = new_beta, updated
        if converged:
            break
    # Refit with the final robust weights before forming its covariance.
    normal = h.conj().T @ (weights[:, None] * h)
    inverse = np.linalg.inv(normal)
    beta = inverse @ (h.conj().T @ (weights * e))
    residual = e - h @ beta
    score = h.conj() * (weights * residual)[:, None]
    score = np.column_stack((score.real, score.imag))
    meat = score.T @ score
    for lag in range(1, min(overlap_lags, len(e) - 1) + 1):
        same = groups[lag:] == groups[:-lag]
        cross = score[lag:][same].T @ score[:-lag][same]
        meat += (1 - lag / (overlap_lags + 1)) * (cross + cross.T)
    bread = np.block([[inverse.real, -inverse.imag], [inverse.imag, inverse.real]])
    covariance = bread @ meat @ bread.T * len(e) / (len(e) - 2)
    covariance = (covariance + covariance.T) / 2
    # Keep each coefficient's 2x2 real/imag covariance for error propagation.
    components = np.stack([covariance[np.ix_([c, c + 2], [c, c + 2])] for c in range(2)])
    return beta, components


def _offdiagonal_products(z, covariance, frequencies, unit_system):
    off = np.column_stack((z[:, 0, 1], z[:, 1, 0]))
    cov = np.stack((covariance[:, 0, 1], covariance[:, 1, 0]), axis=1)
    coefficient = 1 / (2 * np.pi * MU0) if unit_system == "SI" else .2
    rho = coefficient * np.abs(off) ** 2 / frequencies[:, None]
    phase = np.angle(off, deg=True)
    rho_error, phase_error = np.full_like(rho, np.nan), np.full_like(rho, np.nan)
    for c in range(2):
        re_part, im_part = off[:, c].real, off[:, c].imag
        gradient = 2 * coefficient * np.column_stack((re_part, im_part)) / frequencies[:, None]
        variance = np.einsum("ni,nij,nj->n", gradient, cov[:, c], gradient)
        rho_error[:, c] = np.sqrt(np.maximum(variance, 0))
        power = np.abs(off[:, c]) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            gradient = np.column_stack((-im_part, re_part)) / power[:, None]
        variance = np.einsum("ni,nij,nj->n", gradient, cov[:, c], gradient)
        phase_error[:, c] = np.rad2deg(np.sqrt(np.maximum(variance, 0)))
    return rho, rho_error, phase, phase_error


def _native_modules(library_root):
    if library_root is not None:
        root = Path(library_root).resolve()
        if not (root / "razorback").is_dir() or not (root / "razorback_v04").is_dir():
            raise ValueError("library_root must contain razorback and razorback_v04")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        rb = importlib.import_module("razorback")
        rb_v04 = importlib.import_module("razorback_v04")
    except ImportError as exc:
        raise ImportError("Native Razorback backend is unavailable; supply library_root or use backend='scipy'") from exc
    finally:
        sys.dont_write_bytecode = previous
    if library_root is not None:
        for module in (rb, rb_v04):
            if root not in Path(module.__file__).resolve().parents:
                raise ImportError("A different Razorback package is already imported; start a fresh Python process")
    return rb, rb_v04


def _numpy_fourier_coefficients(signal_set, freq, Nper, overlap, window, **kwargs):
    """NumPy frontend equivalent to native per-run time_to_freq.

    Keep the library's discrete-window rules and provided taper exactly. Only
    the execution of its Fourier dot products changes; neither the robust
    estimator nor calibration/segment semantics are replaced.
    """
    if set(kwargs) - {"compute"}:
        raise TypeError("Unsupported Fourier options: " + ", ".join(sorted(kwargs)))
    coefficients, windata, kernels = [], [], {}
    for signal in signal_set.signals:
        # SyncSignal's own module owns the native discrete-window implementation.
        native_module = importlib.import_module(type(signal).__module__)
        native_fourier = native_module.time_to_freq.__module__
        discrete_window = importlib.import_module(native_fourier).discrete_window
        _, length, hop = discrete_window(signal.size, freq / signal.sampling_rate, Nper, overlap)
        key = (signal.sampling_rate, length)
        if key not in kernels:
            taper = window(length) if callable(window) else window
            if taper is not None and np.shape(taper) != (length,):
                raise ValueError("Native taper must match Fourier window length")
            kernel = np.exp(-2j * np.pi * (freq / signal.sampling_rate) * np.arange(length)) * (2 / length)
            if taper is not None:
                kernel *= taper
            kernels[key] = kernel
        kernel = kernels[key]
        count = (signal.size - length) // hop + 1
        batch = max(1, min(4096, 1_000_000 // length))
        local = []
        for channel, calibration in zip(signal.data, signal.calibrations):
            views = np.lib.stride_tricks.sliding_window_view(np.asarray(channel), length)[::hop]
            values = np.empty(count, dtype=complex)
            for start in range(0, count, batch):
                stop = min(start + batch, count)
                values[start:stop] = views[start:stop] @ kernel
            correction = calibration(freq)
            if correction is None:
                raise ValueError("Native calibration returned None")
            local.append(values / correction)
        coefficients.append(local)
        windata.append((count, length, hop))
    if not coefficients:
        raise ValueError("No continuous signal runs")
    result = [np.concatenate(channel) for channel in zip(*coefficients)]
    return result, tuple(zip(*windata))


def _native_estimates(sx, fs_hz, frequencies, intervals, factors, nper, overlap, library_root, fast_fourier):
    rb, rb_v04 = _native_modules(library_root)
    if fast_fourier:
        class NumpyFourierSignalSet(rb.SignalSet):
            fourier_coefficients = _numpy_fourier_coefficients
        signal_set_type = NumpyFourierSignalSet
    else:
        signal_set_type = rb.SignalSet
    lengths = np.ceil(nper * float(fs_hz) / frequencies).astype(int)
    hops = np.maximum(1, (lengths * (1 - overlap)).astype(int))
    z = np.full((len(frequencies), 2, 2), complex(np.nan, np.nan))
    errors = np.full(z.shape, np.nan, dtype=float)
    counts, valid = np.zeros(len(frequencies), int), np.zeros(len(frequencies), bool)
    reasons = []
    exact_fit_frequencies = []
    tags = {"Ex": (0,), "Ey": (1,), "Hx": (2,), "Hy": (3,), "E": (0, 1), "B": (2, 3)}
    # Demean each retained run once, matching build_mt_signal in the reference.
    runs = [(start, stop, np.asarray(sx[:4, start:stop], dtype=float)) for start, stop in intervals]
    runs = [(start, stop, values - values.mean(axis=1, keepdims=True))
            for start, stop, values in runs if np.isfinite(values).all()]
    for i, (frequency, length, hop) in enumerate(zip(frequencies, lengths, hops)):
        # Native discrete_window rejects a run unless it holds at least 2 windows.
        usable = [(start, stop, values) for start, stop, values in runs if stop - start >= length + hop]
        counts[i] = sum((stop - start - length) // hop + 1 for start, stop, _ in usable)
        if counts[i] < 4:
            reasons.append("fewer_than_four_finite_fourier_windows")
            continue
        signals = [rb.SyncSignal(tuple(values), sampling_rate=fs_hz, start=start / fs_hz,
                                 calibrations=tuple(1 / factors[i]))
                   for start, _, values in usable]
        data = signal_set_type(tags, *signals)
        stream = io.StringIO()
        options = dict(Nper=nper, overlap=overlap, window=rb_v04.fourier_transform.slepian_window(2))
        try:
            # Keep native progress output out of CLI JSON/log summaries.
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), \
                    np.errstate(divide="ignore", invalid="ignore"):
                native = rb_v04.utils.impedance(
                    data, [frequency], weights=rb.weights.mest_weights,
                    fourier_opts=options,
                    silent_fail=True)
                if not np.isfinite(native.impedance[0]).all():
                    # Legacy relative-residual convergence can fail when a
                    # noiseless tensor has only floating-point residuals. Prove
                    # an exact full-rank spectral fit before using native OLS;
                    # no noisy/nonconverged robust estimate is substituted.
                    coeffs, _ = data.fourier_coefficients(frequency, **options)
                    h = np.asarray(coeffs[2:4]).T
                    e = np.asarray(coeffs[:2]).T
                    fit, _, rank, singular = np.linalg.lstsq(h, e, rcond=None)
                    residual = np.linalg.norm(e - h @ fit, axis=0)
                    scale = np.maximum(np.linalg.norm(e, axis=0), np.finfo(float).tiny)
                    if (rank == 2 and singular[-1] > singular[0] * 1e-8
                            and np.all(residual / scale <= 1e-12)):
                        native = rb_v04.utils.impedance(
                            data, [frequency], weights=rb.weights.least_square,
                            fourier_opts=options, silent_fail=True)
                        if np.isfinite(native.impedance[0]).all():
                            exact_fit_frequencies.append(float(frequency))
            z[i] = native.impedance[0]
            errors[i] = np.abs(native.error[0])
            valid[i] = np.isfinite(z[i]).all()
            reasons.append("ok" if valid[i] else "native_regression_failed")
        except (ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            reasons.append("native_regression_failed:" + type(exc).__name__)
    metadata = dict(method="razorback_v04.utils.impedance", backend="razorback",
                    library_root=str(Path(rb.__file__).resolve().parent.parent),
                    razorback_version=getattr(rb, "__version__", None),
                    razorback_v04_version=getattr(rb_v04, "__version__", None),
                    weights="razorback.weights.mest_weights", detrend="per retained run channel mean",
                    fourier_backend="numpy_native_equivalent" if fast_fourier else "native_dask",
                    numerically_exact_fit_native_OLS_frequencies=exact_fit_frequencies,
                    standard_errors="native transfer_error (Egbert/Booker); conditional on selected spectra")
    return z, errors, counts, valid, reasons, lengths, hops, metadata


def estimate_impedance(sx, fs_hz, frequencies, intervals=None, calibration=None, *,
                       unit_system="source", backend="scipy", library_root=None, overlap=.71, nper=4,
                       fast_fourier=True):
    """Estimate E = Z H with independent Fourier windows within each interval.

    ``sx`` is (channels, samples), first four channels Ex/Ey/Hx/Hy. Intervals
    use original sample coordinates. Explicit adjacent and separated intervals
    retain their boundaries; Fourier windows cover >=nper periods with
    Slepian NW=2. The default 71% overlap matches the supplied Way2 reference.
    backend='razorback' uses the native robust library and transfer_error.
    backend='scipy' uses an independent Huber complex regression/HAC covariance.
    fast_fourier=True keeps the native estimator but executes the identical
    per-run Fourier projections in bounded NumPy batches instead of Dask.

    unit_system='SI' assumes stored E is V/m and H is A/m, so rho=|Z|²/(2πfμ0).
    'field' assumes (mV/km)/nT and rho=.2|Z|²/f. Neither requires a calibration
    file. 'source' labels the same .2|Z|²/f quantity as an uncalibrated proxy.
    These assumptions are recorded verbatim in unit_assumption metadata.

    Calibration requires positive ``electric_to_mV_per_km``/``magnetic_to_nT``
    scalars. Optional ``channel_factors`` multiply the Fourier coefficients,
    and may be complex (4,) or (frequency,4), or a mapping by channel. JSON
    complex arrays can be encoded as {real: [...], imag: [...]}. An optional
    ``frequency_hz`` grid uses complex Cartesian linear interpolation, without
    extrapolation. Factors must be the complete response correction needed to
    convert stored amplitudes, rather than an instrument forward response.

    z_error is sqrt(var(real Z)+var(imag Z)); z_covariance stores the full 2x2
    real/imag covariance. Propagated rho/phase errors use that covariance.
    These conditional regression standard errors use Bartlett HAC for overlap;
    they are not confidence intervals across independent acquisitions.
    """
    sx = np.asarray(sx)
    if sx.ndim != 2 or sx.shape[0] < 4 or not np.issubdtype(sx.dtype, np.number) or np.iscomplexobj(sx):
        raise ValueError("sx must be a real numeric (at least 4 channels, samples) array")
    if not np.isscalar(fs_hz) or not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    unit_system = "SI" if str(unit_system).lower() == "si" else str(unit_system).lower()
    if unit_system not in {"SI", "field", "source"}:
        raise ValueError("unit_system must be 'SI', 'field', or 'source'")
    if backend not in {"scipy", "razorback"}:
        raise ValueError("backend must be 'scipy' or 'razorback'")
    if not np.isscalar(overlap) or not np.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be finite and in [0,1)")
    if not np.isscalar(nper) or not np.isfinite(nper) or nper < 2:
        raise ValueError("nper must be finite and at least 2")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be a nonempty finite one-dimensional array")
    if np.any(frequencies <= 0) or np.any(frequencies >= fs_hz / 2):
        raise ValueError("frequencies must lie strictly between zero and Nyquist")
    intervals = merge_intervals(intervals, sx.shape[1])
    factors, calibrated = _calibration_factors(calibration, frequencies)
    if calibrated:
        if unit_system == "source":
            unit_system = "field"
        elif unit_system == "SI":
            factors[:, :2] *= 1e-6
            factors[:, 2:] *= 1e-9 / MU0
    n = len(frequencies)
    z = np.full((n, 2, 2), complex(np.nan, np.nan))
    covariance = np.full((n, 2, 2, 2, 2), np.nan)
    counts, valid = np.zeros(n, int), np.zeros(n, bool)
    lengths = np.ceil(nper * float(fs_hz) / frequencies).astype(int)
    hops = np.maximum(1, (lengths * (1 - overlap)).astype(int))
    reasons = []
    metadata = dict(method="frequency_local_complex_Huber_IRLS", backend="scipy",
                    standard_errors="conditional real/imag Bartlett HAC; no record-level inference",
                    detrend="per Fourier-window channel mean")
    if backend == "razorback":
        z, z_error, counts, valid, reasons, lengths, hops, metadata = _native_estimates(
            sx, fs_hz, frequencies, intervals, factors, nper, overlap, library_root, fast_fourier)
    for i, (frequency, length, hop) in enumerate(zip(frequencies, lengths, hops)):
        if backend == "razorback":
            break
        spectra, groups = _spectra(sx, intervals, int(length), int(hop), frequency, fs_hz)
        counts[i] = len(spectra)
        if counts[i] < 4:
            reasons.append("fewer_than_four_finite_fourier_windows")
            continue
        spectra *= factors[i]
        try:
            for electric in range(2):
                z[i, electric], covariance[i, electric] = _fit_complex(
                    spectra[:, 2:4], spectra[:, electric], groups, int(np.ceil(length / hop)) - 1)
            valid[i] = np.isfinite(z[i]).all()
            reasons.append("ok" if valid[i] else "nonfinite_regression")
        except np.linalg.LinAlgError:
            z[i], covariance[i] = np.nan, np.nan
            reasons.append("rank_deficient_magnetic_spectra")
    rho, rho_error, phase, phase_error = _offdiagonal_products(z, covariance, frequencies, unit_system)
    if backend == "razorback":
        off = np.column_stack((z[:, 0, 1], z[:, 1, 0]))
        error = np.column_stack((z_error[:, 0, 1], z_error[:, 1, 0]))
        with np.errstate(invalid="ignore", divide="ignore"):
            rho_error = rho * 2 * error / np.abs(off)
            phase_error = np.rad2deg(error / np.abs(off))
    else:
        z_error = np.sqrt(np.maximum(np.trace(covariance, axis1=-2, axis2=-1), 0))
    metadata.update(periods=nper, overlap=float(overlap), taper="Slepian", time_bandwidth=2,
                    minimum_windows=4, phase_convention="np.angle, degrees in [-180,180]; no quadrant forcing")
    units = {"SI": "(V/m)/(A/m)", "field": "(mV/km)/nT", "source": "stored_E/stored_H"}
    assumption = ("Explicit input convention: stored E in V/m and H in A/m; direct SI formula, no calibration file required"
                  if unit_system == "SI" else
                  "Explicit input convention: E in mV/km and magnetic channels in nT; direct field formula"
                  if unit_system == "field" else
                  "Stored amplitude units unspecified; rho is a numerical proxy, not physical ohm metres")
    return dict(
        frequency=frequencies.copy(), frequencies=frequencies.copy(), z=z, z_error=z_error,
        z_covariance=covariance, n_windows=counts, valid=valid, invalid_reason=reasons,
        rho=rho, rho_error=rho_error, phase=phase, phase_error=phase_error,
        calibrated=calibrated, unit_system=unit_system, unit_assumption=assumption,
        units=dict(z=units[unit_system],
                   rho="ohm m" if unit_system != "source" else "0.2*abs(stored_E/stored_H)^2/Hz (uncalibrated proxy)",
                   phase="degree", frequency="Hz"),
        offdiagonal_order=["xy", "yx"], intervals=np.asarray(intervals, dtype=int).reshape(-1, 2),
        selected_samples=sum(stop - start for start, stop in intervals),
        window_samples=lengths, hop_samples=hops,
        estimator=metadata)


def read_edi_reference(path):
    """Read EDI frequency/off-diagonal Z blocks without an mtpy dependency.

    EDI variance blocks are interpreted as variances and square-rooted. Missing
    blocks/EDI EMPTY sentinels remain NaN. No frequency interpolation, scaling,
    phase editing or component sign changes are performed.
    """
    blocks, current = {}, None
    content = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    empty = re.search(r"\bEMPTY\s*=\s*([+\-\d.eEdD]+)", content, flags=re.I)
    empty = float(empty.group(1).replace("D", "E").replace("d", "e")) if empty else 1e32
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            match = re.match(r">\s*([A-Za-z][A-Za-z0-9_.]*)", stripped)
            current = match.group(1).upper() if match else None
            if current:
                blocks.setdefault(current, [])
            continue
        if current not in {"FREQ", "ZXYR", "ZXYI", "ZYXR", "ZYXI", "ZXY.VAR", "ZYX.VAR"}:
            continue
        for token in stripped.split("!")[0].replace(",", " ").split():
            try:
                blocks[current].append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                continue
    frequency = np.asarray(blocks.get("FREQ", []), dtype=float)
    if not len(frequency) or not np.isfinite(frequency).all() or np.any(frequency <= 0):
        raise ValueError("EDI reference lacks a valid positive FREQ block")
    z = np.full((len(frequency), 2, 2), complex(np.nan, np.nan))
    error = np.full(z.shape, np.nan)
    for prefix, a, b in (("ZXY", 0, 1), ("ZYX", 1, 0)):
        values = []
        for suffix in ("R", "I", ".VAR"):
            block = np.asarray(blocks.get(prefix + suffix, [np.nan] * len(frequency)), dtype=float)
            if len(block) != len(frequency):
                raise ValueError("EDI " + prefix + suffix + " count differs from FREQ")
            block[np.isclose(block, empty, rtol=1e-8, atol=0) | (np.abs(block) >= 1e30)] = np.nan
            values.append(block)
        z[:, a, b] = values[0] + 1j * values[1]
        error[:, a, b] = np.sqrt(np.where(values[2] >= 0, values[2], np.nan))
    off = np.column_stack((z[:, 0, 1], z[:, 1, 0]))
    return dict(frequency=frequency, frequencies=frequency.copy(), z=z, z_error=error,
                rho=.2 * np.abs(off) ** 2 / frequency[:, None], phase=np.angle(off, deg=True),
                units=dict(z="(mV/km)/nT (EDI convention)", rho="ohm m", phase="degree"),
                source=str(path), offdiagonal_order=["xy", "yx"])
