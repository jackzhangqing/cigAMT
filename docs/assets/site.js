'use strict';

const taskDetails = {
  sferic: { title: 'Window-level sferic identification', rate: '24 kHz · 4 × 240', description: 'Identify informative lightning-generated transients in four-component field windows.', input: 'Four-component observation window', target: 'Sferic / non-sferic window label', metrics: 'AP, F1, precision, recall, MCC; complementary impedance cases', protocol: 'Source-record-disjoint partitions. The decision threshold is selected on validation data and frozen for testing.', image: 'sferic-task.png', caption: 'Window-level sferic identification and representative labeled four-component samples.' },
  classification: { title: 'Noise classification', rate: '150 Hz · 4 × 2,048', description: 'Identify the interference type in a four-component observation, with a separate clean class and clean controls.', input: 'Noisy or clean observation window', target: 'Ten noise types + clean', metrics: 'Noisy-class Macro-F1 and accuracy; all-11-class scores and clean accuracy in the manuscript', protocol: 'Noise-injection Protocols A and C; six whole-window SNR levels from −30 to −5 dB.', image: 'classification.png', caption: 'Noise-classification example on a shared four-component observation. It is a single example, not an aggregate result.' },
  localization: { title: 'Noise localization', rate: '150 Hz · 4 × 2,048', description: 'Locate contaminated time intervals. Evaluate both samplewise overlap and complete event recovery.', input: 'Noisy four-component window', target: 'Shared temporal contamination mask', metrics: 'Point F1, IoU, and event F1; false events on clean controls', protocol: 'One-to-one event matching at temporal IoU ≥ 0.5. Noise duration and SNR are varied.', image: 'localization.png', caption: 'Pointwise predictions and event-level localization on a shared noisy input.' },
  denoising: { title: 'Time-domain denoising', rate: '150 Hz · 4 × 2,048', description: 'Recover a four-component reference waveform from a noisy observation while limiting distortion on clean controls.', input: 'Noisy four-component window', target: 'Four-component reference waveform', metrics: 'NRMSE, output SNR, SNR gain, Pearson r, and clean-control NRMSE; impedance cases', protocol: 'Controlled noise type, active support, and input SNR under both injection protocols.', image: 'denoising.png', caption: 'Time-domain restoration by six model architectures on the same corrupted observation.' },
  frequency: { title: 'Frequency-component denoising', rate: '150 Hz · 4 × 2,048', description: 'Restore low- and high-frequency signal components as time-domain waveforms, rather than predicting a spectrogram.', input: 'Noisy four-component window', target: 'Low-band + high-band time-domain waveforms', metrics: 'Overall and bandwise NRMSE, SNR gain, clean-control NRMSE; impedance cases', protocol: 'A 5 Hz FFT split defines the reference components. DC is assigned to the low band; the two components sum to the reference.', image: 'frequency.png', caption: 'Reconstruction from the predicted low- and high-frequency time-domain components.' },
  forecasting: { title: 'Forecasting', rate: '150 Hz · 4 × 2,048', description: 'Predict a held-out future segment using only the observed prefix and an explicit observation mask.', input: 'Observed prefix + observation mask', target: 'Hidden future suffix', metrics: 'Masked MSE, MAE, NRMSE, and Pearson r', protocol: 'Seven horizons: 10, 24, 32, 36, 64, 128, and 256 samples. Scores use only held-out positions.', image: 'forecasting.png', caption: 'Future-suffix forecasting example. Hidden target values are used only for evaluation.' },
  interpolation: { title: 'Interpolation', rate: '150 Hz · 4 × 2,048', description: 'Recover one internal continuous gap from observations before and after the missing interval.', input: 'Two-sided observed context + mask', target: 'Internal contiguous gap', metrics: 'Masked MSE, MAE, NRMSE, and Pearson r', protocol: 'Five gap lengths: 10, 32, 64, 128, and 256 samples. Only hidden entries contribute to the reported errors.', image: 'interpolation.png', caption: 'Interpolation across a continuous internal gap with two-sided temporal context.' },
  imputation: { title: 'Imputation', rate: '150 Hz · 4 × 2,048', description: 'Fill irregular missing runs using temporal context and the remaining observed electromagnetic components.', input: 'Observed context + observation mask', target: 'Irregular contiguous missing runs', metrics: 'Masked MSE, MAE, NRMSE, and Pearson r', protocol: 'Nominal missingness from 10% to 90%, mean runs of 16/32/64/128 samples, and synchronous or channel-independent masks.', image: 'imputation.png', caption: 'Imputation of irregular missing runs across synchronized electromagnetic components.' }
};

const $ = (selector) => document.querySelector(selector);
const taskIds = Object.keys(taskDetails);
let selectedTask = 'sferic';
let benchmarkData = null;
let figureOpener = null;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function announce(text) { $('#announce').textContent = text; }

function selectTask(id, focus = false) {
  if (!taskDetails[id]) return;
  selectedTask = id;
  const task = taskDetails[id];
  document.querySelectorAll('.task-tab').forEach((tab) => {
    const active = tab.dataset.task === id;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    if (active && focus) tab.focus();
  });
  $('#task-panel').setAttribute('aria-labelledby', `tab-${id}`);
  for (const [key, value] of Object.entries({ title: task.title, rate: task.rate, description: task.description, input: task.input, target: task.target, metrics: task.metrics, protocol: task.protocol })) {
    $(`#task-${key}`).textContent = value;
  }
  $('#task-image').src = `assets/images/${task.image}`;
  $('#task-image').alt = task.caption;
  $('#task-image').removeAttribute('width');
  $('#task-image').removeAttribute('height');
  $('#task-image-button').dataset.image = `assets/images/${task.image}`;
  $('#task-image-button').dataset.caption = task.caption;
}

document.querySelectorAll('.task-tab').forEach((tab) => {
  tab.addEventListener('click', () => selectTask(tab.dataset.task));
  tab.addEventListener('keydown', (event) => {
    const index = taskIds.indexOf(tab.dataset.task);
    let next;
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') next = (index + 1) % taskIds.length;
    if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') next = (index - 1 + taskIds.length) % taskIds.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = taskIds.length - 1;
    if (next !== undefined) { event.preventDefault(); selectTask(taskIds[next], true); }
  });
});

function openFigure(button) {
  figureOpener = button;
  $('#dialog-image').src = button.dataset.image;
  $('#dialog-image').alt = button.dataset.caption;
  $('#figure-dialog-caption').textContent = button.dataset.caption;
  $('#open-original').href = button.dataset.image;
  $('#figure-dialog').showModal();
  $('#close-figure').focus();
}

document.querySelectorAll('.figure-zoom').forEach((button) => button.addEventListener('click', () => openFigure(button)));
$('#close-figure').addEventListener('click', () => $('#figure-dialog').close());
$('#figure-dialog').addEventListener('click', (event) => { if (event.target === $('#figure-dialog')) $('#figure-dialog').close(); });
$('#figure-dialog').addEventListener('close', () => { if (figureOpener) figureOpener.focus(); });

function currentResultTask() { return benchmarkData?.tasks.find((task) => task.id === $('#results-task').value); }

function fillSortOptions(task) {
  const select = $('#results-sort');
  select.replaceChildren(new Option('Manuscript order', ''));
  task.metrics.forEach((metric) => select.add(new Option(`${metric.label} ${metric.direction === 'max' ? '↑' : '↓'}`, metric.key)));
}

function visibleRows(task) {
  const rows = [...task.rows];
  const metric = task.metrics.find((item) => item.key === $('#results-sort').value);
  if (metric) rows.sort((a, b) => {
    const av = a.values[metric.key]?.mean, bv = b.values[metric.key]?.mean;
    if (av == null) return 1;
    if (bv == null) return -1;
    return metric.direction === 'max' ? bv - av : av - bv;
  });
  return rows;
}

function renderResults(resetSort = false) {
  const task = currentResultTask();
  if (!task) return;
  if (resetSort) fillSortOptions(task);
  const head = $('#results-table thead'), body = $('#results-table tbody');
  head.replaceChildren(); body.replaceChildren();
  const header = document.createElement('tr');
  const modelHeader = document.createElement('th'); modelHeader.scope = 'col'; modelHeader.textContent = 'Model'; header.append(modelHeader);
  const best = {};
  task.metrics.forEach((metric) => {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = `${metric.label} ${metric.direction === 'max' ? '↑' : '↓'}`;
    if (metric.unit) { const unit = document.createElement('span'); unit.className = 'metric-unit'; unit.textContent = metric.unit; th.append(unit); }
    header.append(th);
    const values = task.rows.map((row) => row.values[metric.key]?.mean).filter((value) => Number.isFinite(value));
    best[metric.key] = metric.direction === 'max' ? Math.max(...values) : Math.min(...values);
  });
  head.append(header);
  visibleRows(task).forEach((row) => {
    const tr = document.createElement('tr');
    const model = document.createElement('td'); model.textContent = row.model; tr.append(model);
    task.metrics.forEach((metric) => {
      const td = document.createElement('td');
      const value = row.values[metric.key];
      if (value?.mean == null) td.textContent = '—';
      else {
        const mean = document.createElement('span'); mean.textContent = value.mean.toFixed(metric.decimals); td.append(mean);
        if (Math.abs(value.mean - best[metric.key]) < 1e-10) { td.classList.add('best'); td.title = 'Best reported mean in this comparison'; }
        if (value.sd != null) { const sd = document.createElement('span'); sd.className = 'dispersion'; sd.textContent = ` ± ${value.sd.toFixed(metric.decimals)}`; td.append(sd); }
      }
      tr.append(td);
    });
    body.append(tr);
  });
  $('#results-description').textContent = `${task.description} Source: ${task.sourceTable}.`;
  $('#results-note').textContent = task.branchHz === 24000
    ? 'Sferic AP, F1, precision, and recall are percentages; MCC is multiplied by 100. Values are mean ± reported sample SD. Highlights indicate the best mean in the displayed comparison.'
    : 'AMT values use the manuscript’s task-specific units. A/C noise-task summaries follow the manuscript; reported dispersions are descriptive and do not establish pooled uncertainty. Highlights indicate the best displayed mean, including deterministic controls where present.';
}

$('#results-task').addEventListener('change', () => renderResults(true));
$('#results-sort').addEventListener('change', () => renderResults());
$('#task-show-results').addEventListener('click', () => {
  $('#results-task').value = selectedTask;
  renderResults(true);
  $('#results').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth' });
  announce(`Showing ${taskDetails[selectedTask].title} baseline results.`);
});

function csvField(value) { return `"${String(value).replaceAll('"', '""')}"`; }

$('#download-results').addEventListener('click', () => {
  const task = currentResultTask(); if (!task) return;
  const header = ['model', ...task.metrics.flatMap((metric) => [`${metric.key}_mean`, `${metric.key}_sd`])];
  const rows = visibleRows(task).map((row) => [row.model, ...task.metrics.flatMap((metric) => {
    const value = row.values[metric.key];
    return [value?.mean == null ? '' : value.mean.toFixed(metric.decimals), value?.sd == null ? '' : value.sd.toFixed(metric.decimals)];
  })]);
  const blob = new Blob(['\ufeff' + [header, ...rows].map((row) => row.map(csvField).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = `cigamt_${task.id}_manuscript_results.csv`; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$('#copy-bibtex').addEventListener('click', async () => {
  const text = $('#bibtex code').textContent;
  try {
    await navigator.clipboard.writeText(text);
    $('#copy-bibtex').textContent = 'Copied'; announce('BibTeX copied.');
    setTimeout(() => { $('#copy-bibtex').textContent = 'Copy BibTeX'; }, 1800);
  } catch {
    const selection = window.getSelection(); const range = document.createRange(); range.selectNodeContents($('#bibtex code')); selection.removeAllRanges(); selection.addRange(range);
    announce('Select and copy the highlighted BibTeX text.'); $('#copy-bibtex').textContent = 'Text selected';
  }
});

function validPublicUrl(value) {
  if (typeof value !== 'string') return false;
  try { return new URL(value).protocol === 'https:'; } catch { return false; }
}

function configureLink(id, url, label) {
  const link = $(id); if (!link || !validPublicUrl(url)) return;
  link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = `${label} ↗`;
}

async function loadConfiguration() {
  try {
    const response = await fetch('site-config.json'); if (!response.ok) return;
    const config = await response.json();
    if (config.repositoryPublished) configureLink('#repo-link', config.repositoryUrl, 'Project repository');
    else { $('#repo-link').href = '#resources'; $('#repo-link').removeAttribute('target'); $('#repo-link').textContent = 'Repository launch in preparation'; }
    configureLink('#paper-link', config.paperUrl, 'Read the paper');
    configureLink('#dataset-link', config.datasetUrl, 'Dataset on Hugging Face');
    configureLink('#code-link', config.benchmarkCodeUrl, 'Browse benchmark code');
    configureLink('#weights-link', config.weightsUrl, 'GitHub Releases');
    configureLink('#inventory-link', config.inventoryUrl, 'Read the data inventory');
  } catch { /* Static availability text remains truthful if configuration cannot load. */ }
}

async function loadResults() {
  try {
    const response = await fetch('data/benchmarks.json'); if (!response.ok) throw new Error(`HTTP ${response.status}`);
    benchmarkData = await response.json();
    if (!Array.isArray(benchmarkData.tasks) || benchmarkData.tasks.length !== 8) throw new Error('Invalid task inventory');
    renderResults(true);
  } catch {
    $('#results-description').classList.add('table-error');
    $('#results-description').textContent = 'Results could not be loaded. Open this site through a local HTTP server or GitHub Pages, or download data/benchmarks.json directly.';
    $('#download-results').disabled = true;
  }
}

loadConfiguration();
loadResults();
