"""Paired impedance errors on one strict, all-method common frequency mask.

Relative complex-impedance RMSE is an error metric, not a relative-impedance
replacement for the apparent-resistivity curves. All rho arrays remain in the
configured physical convention. No pseudo ground truth is assigned to fields.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

MODELS = ('tcn','unet1d','bigru','patch_transformer','amt_maskformer','spectral_fno')
METHODS = ('denoising','frequency_denoising')
METRICS = ('relative_z_rmse','rho_log10_rmse','phase_mae_deg')
COMPONENTS = (('xy',0,1),('yx',1,0))


def _frequencies(estimate):
    key = 'frequency' if 'frequency' in estimate else 'frequencies_hz'
    f = np.asarray(estimate[key],dtype=float)
    if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or np.any(f<=0):
        raise ValueError('Frequency grid must be a finite positive nonempty vector')
    return f


def valid_frequency_mask(estimate):
    """Accepted frequencies with finite nonzero off-diagonal Z and positive rho."""
    f = _frequencies(estimate)
    z, rho = np.asarray(estimate['z']), np.asarray(estimate['rho'],dtype=float)
    accepted = np.asarray(estimate['accepted'],dtype=bool)
    if z.shape != (len(f),2,2) or rho.shape != z.shape or accepted.shape != f.shape:
        raise ValueError('Invalid estimate array geometry')
    valid = accepted.copy()
    for _,a,b in COMPONENTS:
        valid &= np.isfinite(z[:,a,b]) & np.isfinite(rho[:,a,b]) & (rho[:,a,b]>0) & (abs(z[:,a,b])>0)
    return valid


def _same_grid(reference,candidate):
    a,b = _frequencies(reference),_frequencies(candidate)
    if not np.array_equal(a,b):
        raise ValueError('All methods must use exactly the same requested frequency grid')
    return a


def compare(reference,candidate,common_mask=None):
    """Return equal-component-mean errors over shared xy/yx frequency support.

    For component k, relative_z_rmse_k = sqrt(sum|Zhat-Z|^2 / sum|Z|^2).
    rho_log10_rmse_k is RMSE(log10(rhohat)-log10(rho)) in decades.
    phase_mae_deg_k averages abs(wrap180(angle(Zhat)-angle(Z))) in degrees.
    Each combined metric is the arithmetic mean of its xy and yx values. It is
    not an energy-weighted or joint-tensor normalization across components.
    """
    f = _same_grid(reference,candidate)
    valid = valid_frequency_mask(reference) & valid_frequency_mask(candidate)
    if common_mask is None:
        mask = valid
    else:
        mask = np.asarray(common_mask,dtype=bool)
        if mask.shape != f.shape:
            raise ValueError('Common mask shape differs from frequency grid')
        # Silently changing a caller's common support could make methods unfair.
        if np.any(mask & ~valid):
            raise ValueError('Common mask includes an invalid reference/candidate frequency')
    n = int(mask.sum())
    result = {'n_frequencies':n, 'frequencies_hz':f[mask].tolist(),
              'aggregation':'mean of componentwise xy/yx errors; equal frequency weights within each component'}
    for metric in METRICS:
        result[metric] = None
        result[metric+'_xy'] = None
        result[metric+'_yx'] = None
    if n == 0:
        return result
    for name,a,b in COMPONENTS:
        target = np.asarray(reference['z'])[mask,a,b]
        prediction = np.asarray(candidate['z'])[mask,a,b]
        energy = float(np.sum(abs(target)**2))
        if energy <= 0:
            raise ValueError('Reference impedance energy is zero on common support')
        result['relative_z_rmse_'+name] = float(np.sqrt(np.sum(abs(prediction-target)**2)/energy))
        rt = np.asarray(reference['rho'],dtype=float)[mask,a,b]
        rp = np.asarray(candidate['rho'],dtype=float)[mask,a,b]
        result['rho_log10_rmse_'+name] = float(np.sqrt(np.mean((np.log10(rp)-np.log10(rt))**2)))
        delta = np.angle(prediction,deg=True)-np.angle(target,deg=True)
        wrapped = (delta+180.0)%360.0-180.0
        result['phase_mae_deg_'+name] = float(np.mean(abs(wrapped)))
    for metric in METRICS:
        result[metric] = float(np.mean([result[metric+'_xy'],result[metric+'_yx']]))
    return result


def _entries(estimates,models):
    """Canonical structure: input plus method->{model:estimate} dictionaries."""
    if 'input' not in estimates:
        raise KeyError('estimates requires input')
    entries = [('input','input',estimates['input'])]
    for method in METHODS:
        if method not in estimates:
            raise KeyError('Missing prediction method: '+method)
        for model in models:
            if model not in estimates[method]:
                raise KeyError('Missing trained prediction %s/%s' % (method,model))
            entries.append((model,method,estimates[method][model]))
    return entries


def common_frequency_mask(estimates,models=MODELS,include_reference=True):
    """Intersect reference, input and all configured denoising/frequency outputs."""
    entries = _entries(estimates,tuple(models))
    anchor = estimates['input']
    mask = valid_frequency_mask(anchor)
    if include_reference:
        reference = estimates.get('reference')
        if reference is None:
            raise ValueError('Reference is required for quantitative test errors')
        _same_grid(anchor,reference)
        mask &= valid_frequency_mask(reference)
    for _,_,estimate in entries:
        _same_grid(anchor,estimate)
        mask &= valid_frequency_mask(estimate)
    return mask


def score_case(case,estimates):
    """Return 13 rows: raw input and six models times two task outputs.

    case['has_reference'] must be explicit. For measured fields with no clean
    truth, errors are None; only estimator coverage/QC is recorded. Never use
    the original field as clean ground truth. A zero common mask remains zero.
    """
    if 'has_reference' not in case:
        raise ValueError('case.has_reference must be explicit')
    models = tuple(case.get('models',MODELS))
    if not models or len(models)!=len(set(models)):
        raise ValueError('models must be a nonempty unique list')
    has_reference = bool(case['has_reference'])
    common = common_frequency_mask(estimates,models,include_reference=has_reference)
    f = _frequencies(estimates['input'])
    reference = estimates.get('reference')
    ref_valid_n = int(valid_frequency_mask(reference).sum()) if has_reference else None
    rows = []
    for model,method,estimate in _entries(estimates,models):
        individual_n = int(valid_frequency_mask(estimate).sum())
        row = {'case_id':str(case.get('case_id',case.get('id','unknown'))),
               'case_kind':case.get('kind','test' if has_reference else 'field'),
               'has_reference':has_reference, 'model':model, 'method':method,
               'requested_frequency_n':len(f), 'individual_valid_n':individual_n,
               'individual_coverage':individual_n/len(f),
               'reference_valid_n':ref_valid_n,
               'strict_common_n':int(common.sum()),
               'strict_common_frequency_hz':f[common].tolist(),
               'metric_support':'same reference/input/all-model/all-method frequency intersection',
               'reference_type':'clean test target' if has_reference else 'no clean ground truth',
               'impedance_unit_convention':estimate.get('metadata',{}).get('unit_convention')}
        for k in ('split','scheme','snr_db','snr_target_db','class_id','noise_length','source_id','source_row'):
            if k in case: row[k] = case[k]
        if has_reference:
            values = compare(reference,estimate,common)
            row.update({k:v for k,v in values.items() if k not in ('frequencies_hz','n_frequencies')})
        else:
            for metric in METRICS:
                row[metric] = row[metric+'_xy'] = row[metric+'_yx'] = None
            row['aggregation'] = 'No quantitative truth-error score for original measured field'
        rows.append(row)
    return rows


def summarize_test_cases(rows,expected_cases=None):
    """Equal-case means over preselected referenced test cases, with counts.

    Frequencies and samples are never pooled across cases; fields are excluded.
    Undefined cases remain missing and cannot create a zero error or a CI.
    """
    groups = defaultdict(list)
    for row in rows:
        if row.get('has_reference'):
            groups[(row['model'],row['method'])].append(row)
    result = []
    for (model,method),records in groups.items():
        case_ids = [r['case_id'] for r in records]
        if len(case_ids)!=len(set(case_ids)):
            raise ValueError('Duplicate case rows would overweight a test case: %s/%s' % (model,method))
        item = {'model':model,'method':method,'n_cases_total':len(records),
                'expected_n_cases':expected_cases,
                'case_aggregation':'arithmetic mean of per-case errors; no frequency pooling or confidence interval'}
        for metric in METRICS:
            values = [float(r[metric]) for r in records if r.get(metric) is not None and np.isfinite(r[metric])]
            item[metric+'_mean'] = float(np.mean(values)) if values else None
            item[metric+'_n_cases'] = len(values)
        item['n_cases_scored'] = sum(all(r.get(m) is not None and np.isfinite(r[m]) for m in METRICS) for r in records)
        item['strict_common_n_per_case'] = {r['case_id']:r['strict_common_n'] for r in records}
        result.append(item)
    return result


summarize = summarize_test_cases
