"""Coverage-first, paired A/C noise schedules for the fixed epoch-zero export.

Each split keeps the factory's order: one clean row, all SNR conditions, next
clean row. Noise domains rotate as (clean_position + SNR_position) % n_domains.
If a domain has enough assigned slots, every eligible template appears at least
once. Remaining slots balance available (type,length) cells. When slots are
insufficient, a capacity-constrained cell allocation maximizes unique coverage.
The resulting selections are shuffled within each domain before assignment.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Any

import numpy as np

from .utils import SPLITS, digest, file_sha, json_read, json_write, seed64


def _eligible_rows(noise_index: dict[str, np.ndarray]) -> np.ndarray:
    n = len(noise_index['row_id'])
    for key in ('domain_id', 'type_id', 'length', 'split_id', 'crop_start'):
        if noise_index[key].shape != (n,):
            raise ValueError(f'Invalid noise index column {key}: {noise_index[key].shape}')
    if len(np.unique(noise_index['row_id'])) != n:
        raise ValueError('Noise bank row IDs must be unique')
    if np.any((noise_index['split_id'] < 0) | (noise_index['split_id'] >= len(SPLITS))):
        raise ValueError('Noise index contains an unknown split role')
    if np.any((noise_index['domain_id'] == 2) & (noise_index['split_id'] != 0)):
        raise ValueError('Unverified generated noise must stay train-only')
    eligible = ((noise_index['type_id'] > 0) & (noise_index['length'] >= 2)
                & (noise_index['length'] <= 2048) & (noise_index['crop_start'] >= 0)
                & (noise_index['crop_start'] + noise_index['length'] <= 2048))
    # Production indices contain only fully audited rows. Optional explicit
    # columns allow portable fixtures or future catalogs to reject zero/invalid
    # rows without ever classifying a clean/zero template as interference.
    if 'valid' in noise_index:
        eligible &= noise_index['valid'].astype(bool)
    if 'flags' in noise_index:
        eligible &= noise_index['flags'] == 0
    for key in ('rms', 'crop_rms'):
        if key in noise_index:
            eligible &= np.isfinite(noise_index[key]) & (noise_index[key] > 0)
    return eligible


def _cells(noise_index, rows):
    result = []
    type_ids = noise_index['type_id'][rows]
    lengths = noise_index['length'][rows]
    for type_id in sorted(np.unique(type_ids).tolist()):
        typed = type_ids == type_id
        for length in sorted(np.unique(lengths[typed]).tolist()):
            result.append((int(type_id), int(length), rows[typed & (lengths == length)]))
    return result


def _balanced_allocation(total: int, capacities: np.ndarray | None, n_cells: int,
                         rng: np.random.Generator) -> np.ndarray:
    allocation = np.zeros(n_cells, dtype=np.int64)
    if total == 0:
        return allocation
    if n_cells == 0:
        raise ValueError('Cannot allocate noise draws to an empty cell pool')
    order = rng.permutation(n_cells)
    if capacities is None:
        allocation[:] = total // n_cells
        allocation[order[:total % n_cells]] += 1
        return allocation
    if total > int(capacities.sum()):
        raise ValueError('Without-replacement allocation exceeds available templates')
    remaining = total
    while remaining:
        available = order[allocation[order] < capacities[order]]
        if len(available) == 0:
            raise ValueError('Allocation exhausted before requested quota')
        per_cell = remaining // len(available)
        if per_cell == 0:
            allocation[available[:remaining]] += 1
            break
        additions = np.minimum(capacities[available] - allocation[available], per_cell)
        allocation[available] += additions
        remaining -= int(additions.sum())
    return allocation


def _domain_choices(noise_index, pool: np.ndarray, quota: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    cells = _cells(noise_index, pool)
    available = len(pool)
    if quota >= available:
        # Mandatory coverage, then cell-balanced supplemental draws. Capacity
        # imbalance of the mandatory part is deliberately retained and reported.
        parts = [pool.copy()]
        extra = _balanced_allocation(quota - available, None, len(cells), rng)
        for (_, _, choices), draws in zip(cells, extra):
            if draws:
                cycle = choices[rng.permutation(len(choices))]
                parts.append(np.resize(cycle, int(draws)))
        selected = np.concatenate(parts)
        policy = 'all_templates_once_then_cell_balanced_repeats'
    else:
        allocation = _balanced_allocation(quota, np.asarray([len(c[2]) for c in cells], np.int64),
                                           len(cells), rng)
        parts = []
        for (_, _, choices), draws in zip(cells, allocation):
            if draws:
                parts.append(choices[rng.permutation(len(choices))[:int(draws)]])
        selected = np.concatenate(parts) if parts else np.empty(0, np.int64)
        policy = 'capacity_constrained_cell_balance_without_replacement'
    if len(selected) != quota:
        raise AssertionError('Noise draw allocation does not match domain quota')
    # Mix mandatory coverage and repetitions; do not create a coverage-first
    # temporal block associated with a particular clean source or SNR.
    selected = selected[rng.permutation(len(selected))]
    unique = len(np.unique(selected))
    if quota >= available and unique != available:
        raise AssertionError('A sufficient domain quota failed to cover all templates')
    if quota < available and unique != quota:
        raise AssertionError('An insufficient domain quota unnecessarily repeated a template')
    usage = np.bincount(selected, minlength=len(noise_index['row_id']))
    cell_counts = [{'type_id': type_id, 'length': length,
                    'available_templates': len(choices), 'draws': int(usage[choices].sum())}
                   for type_id, length, choices in cells]
    cell_draws = [c['draws'] for c in cell_counts]
    template_usage = usage[pool]
    return selected, {'available': available, 'unique_used': unique, 'draws': quota,
                      'full_coverage': unique == available,
                      'coverage_fraction': unique / available if available else 1.,
                      'sufficient_slots': quota >= available, 'policy': policy,
                      'available_cells': len(cells), 'type_length_counts': cell_counts,
                      'type_length_draw_min': min(cell_draws) if cell_draws else 0,
                      'type_length_draw_max': max(cell_draws) if cell_draws else 0,
                      'template_repeat_min': int(template_usage.min()) if len(template_usage) else 0,
                      'template_repeat_max': int(template_usage.max()) if len(template_usage) else 0}


def _condition_histogram(noise_index, selected, conditions, snrs, manifest):
    if len(selected) == 0:
        return []
    domains = noise_index['domain_id'][selected].astype(np.int64)
    kinds = noise_index.get('source_kind', noise_index['domain_id'])[selected].astype(np.int64)
    types = noise_index['type_id'][selected].astype(np.int64)
    unique_lengths = np.unique(noise_index['length'][selected])
    length_index = np.searchsorted(unique_lengths, noise_index['length'][selected])
    n_kind = int(kinds.max()) + 1
    n_type = int(types.max()) + 1
    n_length, n_snr = len(unique_lengths), len(snrs)
    codes = ((((domains * n_kind + kinds) * n_type + types) * n_length + length_index) * n_snr + conditions)
    counts = np.bincount(codes)
    class_names = manifest['noise'].get('class_names', [])
    domain_names = manifest['noise'].get('domain_names', ['real', 'synthetic', 'generated'])
    source_names = manifest['noise'].get('source_kind_names', [])
    rows = []
    for code in np.flatnonzero(counts):
        value, condition = divmod(int(code), n_snr)
        value, length = divmod(value, n_length)
        value, type_id = divmod(value, n_type)
        domain, kind = divmod(value, n_kind)
        rows.append({'domain_id': domain, 'domain': domain_names[domain] if domain < len(domain_names) else str(domain),
                     'source_kind': kind, 'source': source_names[kind] if kind < len(source_names) else str(kind),
                     'type_id': type_id, 'noise_type': class_names[type_id] if type_id < len(class_names) else str(type_id),
                     'length': int(unique_lengths[length]), 'snr_condition_index': condition,
                     'snr_db': float(snrs[condition]), 'count': int(counts[code])})
    return rows


def _clean_source_domain_histogram(clean_index, clean_ids, selected, noise_index, conditions, snrs, manifest):
    if len(selected) == 0:
        return []
    sources = clean_index['source_id'][clean_ids].astype(np.int64)
    domains = noise_index['domain_id'][selected].astype(np.int64)
    n_domain = int(domains.max()) + 1
    codes = (sources * n_domain + domains) * len(snrs) + conditions
    counts = np.bincount(codes)
    source_specs = manifest['clean'].get('sources', [])
    rows = []
    for code in np.flatnonzero(counts):
        value, condition = divmod(int(code), len(snrs))
        source, domain = divmod(value, n_domain)
        rows.append({'clean_source_id': source,
                     'clean_source': source_specs[source].get('id', str(source)) if source < len(source_specs) else str(source),
                     'noise_domain_id': domain, 'snr_condition_index': condition,
                     'snr_db': float(snrs[condition]), 'count': int(counts[code])})
    return rows


def prepare_schedule(root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Create deterministic epoch-zero schedules, shared by schemes A and C.

    NPZ ``noise_index`` indexes rows in noise/noise_index.npz, not original file
    row numbers. ``draw_seed`` is the channel-perturbation seed used by the
    existing SampleFactory result. ``template_draw_seed`` retains the preceding
    seed64(dataset_id,'noise',split,clean_index,SNR_condition,0) derivation.
    ``noise_start`` is drawn from that preceding seed and uses the selected
    support length. Dynamic epochs >0 continue the stochastic factory protocol.
    """
    root = Path(root).resolve()
    metadata = root / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    clean_path = metadata / 'clean_index.npz'
    noise_path = root / manifest['noise']['pool_arrays_path']
    with np.load(clean_path, allow_pickle=False) as z:
        clean_index = {k: z[k].copy() for k in z.files}
    with np.load(noise_path, allow_pickle=False) as z:
        noise_index = {k: z[k].copy() for k in z.files}
    snrs = np.asarray(manifest['config']['noise']['snr_db'], dtype=np.float64)
    if snrs.ndim != 1 or len(snrs) == 0 or not np.isfinite(snrs).all() or len(np.unique(snrs)) != len(snrs):
        raise ValueError('SNR conditions must be a nonempty unique finite list')
    if clean_index['split_id'].ndim != 1 or len(clean_index['source_id']) != len(clean_index['split_id']):
        raise ValueError('Invalid clean index layout')
    if np.any((clean_index['split_id'] < 0) | (clean_index['split_id'] >= len(SPLITS))):
        raise ValueError('Clean index contains unknown split IDs')
    eligible = _eligible_rows(noise_index)
    signature = digest({'version': 1, 'dataset_id': manifest['dataset_id'],
                        'clean_index_sha256': file_sha(clean_path), 'noise_index_sha256': file_sha(noise_path),
                        'snr_db': snrs.tolist(), 'epoch': 0,
                        'policy': 'Latin_domains_all_templates_then_cell_balance_shuffle',
                        'seed_semantics': 'draw_seed=channel_perturbation; template_draw_seed=initial'})
    report_path = metadata / 'noise_coverage_schedule_report.json'
    if report_path.is_file():
        old = json_read(report_path)
        if old.get('signature') == signature and all((root / r['path']).is_file()
                and file_sha(root / r['path']) == r['sha256'] for r in old['splits'].values()):
            print('NOISE COVERAGE SCHEDULE REUSE', flush=True)
            return old
        if old.get('signature') != signature:
            raise ValueError('Existing noise coverage schedules belong to different sources or settings')
    begin = time.monotonic()
    result = {'schema_version': 1, 'signature': signature, 'dataset_id': manifest['dataset_id'],
              'epoch': 0, 'schemes_shared': ['A', 'C'], 'snr_db': snrs.tolist(),
              'report_path': 'metadata/noise_coverage_schedule_report.json',
              'eligible_template_count': int(eligible.sum()),
              'excluded_index_rows': int(np.count_nonzero(~eligible)),
              'eligibility': 'fully audited noise bank; optional validity/flags/raw-RMS columns honored',
              'domain_rule': '(clean_position_in_split + SNR_condition_index) % number_of_available_domains',
              'coverage_rule': 'all templates once when quota suffices; remaining draws balance available type/length cells',
              'balancing_limit': 'mandatory full coverage preserves unequal source/type/length pool sizes; full joint uniformity is not claimed',
              'seed_rule': 'initial=seed64(dataset_id,noise,split,global_clean_index,SNR_condition,0); draw_seed=seed64(initial,channel_perturbation)',
              'dynamic_epochs': 'This table is epoch0 only; later epochs use the stochastic factory sampling rules',
              'splits': {}}
    all_used = []
    for role, split in enumerate(SPLITS):
        split_clean = np.flatnonzero(clean_index['split_id'] == role)
        n_clean = len(split_clean)
        count = n_clean * len(snrs)
        clean_ids = np.repeat(split_clean, len(snrs))
        base_position = np.repeat(np.arange(n_clean, dtype=np.int64), len(snrs))
        conditions = np.tile(np.arange(len(snrs), dtype=np.int64), n_clean)
        role_eligible = eligible & (noise_index['split_id'] == role)
        domains = sorted(np.unique(noise_index['domain_id'][role_eligible]).tolist())
        if count and not domains:
            raise ValueError(f'No eligible noise exists for clean split {split}')
        selected = np.full(count, -1, dtype=np.int64)
        starts = np.zeros(count, dtype=np.int16)
        seeds = np.zeros(count, dtype=np.uint64)
        initial_seeds = np.zeros(count, dtype=np.uint64)
        domain_reports = []
        if count:
            domain_position = (base_position + conditions) % len(domains)
            for domain_position_id, domain in enumerate(domains):
                positions = np.flatnonzero(domain_position == domain_position_id)
                pool = np.flatnonzero(role_eligible & (noise_index['domain_id'] == domain))
                rng = np.random.default_rng(seed64(manifest['dataset_id'], 'noise_coverage', split, int(domain), 0))
                choices, details = _domain_choices(noise_index, pool, len(positions), rng)
                selected[positions] = choices
                per_snr = np.bincount(conditions[positions], minlength=len(snrs))
                details.update(domain_id=int(domain),
                               domain=manifest['noise'].get('domain_names', ['real', 'synthetic', 'generated'])[int(domain)],
                               snr_counts=[{'snr_db': float(snr), 'count': int(value)} for snr, value in zip(snrs, per_snr)],
                               per_snr_max_minus_min=int(per_snr.max() - per_snr.min()))
                domain_reports.append(details)
            if np.any(selected < 0) or not np.all(eligible[selected]) or not np.all(noise_index['split_id'][selected] == role):
                raise AssertionError('Schedule selected an ineligible or wrong-split noise template')
            lengths = noise_index['length'][selected]
            for i, (clean_id, condition, length) in enumerate(zip(clean_ids, conditions, lengths)):
                draw = seed64(manifest['dataset_id'], 'noise', split, int(clean_id), int(condition), 0)
                initial_seeds[i] = draw
                starts[i] = int(np.random.default_rng(draw).integers(2048 - int(length) + 1))
                seeds[i] = seed64(draw, 'channel_perturbation')
                if (i + 1) % 250000 == 0:
                    print(f'NOISE SCHEDULE {split} seeds {i+1}/{count}', flush=True)
            if np.any(starts < 0) or np.any(starts.astype(np.int64) + lengths > 2048):
                raise AssertionError('Noise event placement exceeds the signal clip')
        path = metadata / f'noise_schedule_{split}.npz'
        partial = path.with_name(path.name + '.partial')
        with partial.open('wb') as stream:
            np.savez_compressed(stream, noise_index=selected, noise_start=starts,
                                draw_seed=seeds, template_draw_seed=initial_seeds)
        partial.replace(path)
        used = np.unique(selected)
        used = used[used >= 0]
        all_used.append(used)
        descriptor = {'path': str(path.relative_to(root)).replace('\\', '/'),
                      'schedulepath': str(path.relative_to(root)).replace('\\', '/'),
                      'sha256': file_sha(path), 'count': count, 'clean_count': n_clean,
                      'paired_A_C_rows': count * 2, 'domains': domain_reports,
                      'unique_used': len(used), 'available': int(role_eligible.sum()),
                      'full_coverage': len(used) == int(role_eligible.sum()),
                      'source_type_length_snr_histogram': _condition_histogram(noise_index, selected, conditions, snrs, manifest),
                      'clean_source_domain_snr_histogram': _clean_source_domain_histogram(clean_index, clean_ids, selected,
                                                                                        noise_index, conditions, snrs, manifest)}
        result['splits'][split] = descriptor
        print(f'NOISE SCHEDULE {split} {count} draws, {len(used)}/{int(role_eligible.sum())} unique eligible templates', flush=True)
    unique_all = np.unique(np.concatenate(all_used)) if all_used else np.empty(0, np.int64)
    result.update(total_scheduled_draws_per_scheme=sum(s['count'] for s in result['splits'].values()),
                  total_scheduled_A_C_rows=sum(s['paired_A_C_rows'] for s in result['splits'].values()),
                  unique_used_all_splits=len(unique_all),
                  full_coverage_all_splits=len(unique_all) == int(eligible.sum()),
                  elapsed_seconds=time.monotonic() - begin)
    json_write(report_path, result)
    return result
