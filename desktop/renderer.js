// Window logic. All data reaches the DOM through textContent, never innerHTML.
const api = window.inspector;
const $ = (id) => document.getElementById(id);
const STATUS_LABELS = { complete: 'OK', review_required: 'Check', reviewed: 'Checked', skipped: 'Skipped' };
const TRAY_NAMES = { bare: 'Bare', packaged: 'Packaged' };
const MAX_LOG_LINES = 500;
const SVG = 'http://www.w3.org/2000/svg';

const state = {
  tray: 'auto', filter: 'all', results: [], total: 0, selected: null, userPicked: false,
  phase: 'idle', // idle | running | done | stopped | failed
  error: '', cancelled: false, source: null, outputDir: '', worker: { status: 'loading' },
  editing: false, pending: new Map(), // "row,column" -> new label, unsaved
  enhance: { bare: true, packaged: false }, // display-only contrast stretch, remembered per tray type
};

const cleanError = (error) => error.message.replace(/^Error invoking remote method '[^']+': (Error: )?/, '');
const slotKey = (slot) => `${slot.row},${slot.column}`;
const effectiveLabel = (slot) => state.pending.get(slotKey(slot)) ?? slot.label;
const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;
const baseName = (fullPath) => fullPath.split(/[\\/]/).filter(Boolean).pop() ?? fullPath;
const byStatus = (status) => state.results.filter((result) => result.status === status);

function svgElement(name, attributes) {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  return element;
}

function pressOne(container, button) {
  container.querySelectorAll('button').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
}

function appendLog(line) {
  const log = $('log');
  const entry = document.createElement('div');
  entry.textContent = line;
  log.append(entry);
  while (log.childElementCount > MAX_LOG_LINES) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

let toastTimer = null;
function toast(text) {
  $('toast').textContent = text;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}

function cell(text, className) {
  const td = document.createElement('td');
  if (className) td.className = className;
  td.textContent = text ?? '–';
  return td;
}

function chipCell(text, className) {
  const td = document.createElement('td');
  const chip = document.createElement('span');
  chip.className = `chip ${className}`;
  chip.textContent = text;
  td.append(chip);
  return td;
}

function stopEditing() {
  state.editing = false;
  state.pending.clear();
}

function confirmDiscard() {
  return !state.pending.size || window.confirm('You have changes that are not saved. Throw them away?');
}

function select(result, byUser) {
  if (result === state.selected) return;
  if (!confirmDiscard()) return;
  stopEditing();
  state.selected = result;
  if (byUser) state.userPicked = true;
  render();
}

// ---------- Status banner ----------

function renderBanner() {
  const count = state.results.length;
  const toCheck = byStatus('review_required').length;
  const skipped = byStatus('skipped').length;
  let tone = 'idle';
  let title = 'Choose a folder or image to start';
  let detail = 'Use “Inspect a folder” on the left.';
  const show = { progress: false, stop: false, check: false, output: false, details: false };

  if (state.phase === 'running') {
    tone = 'busy';
    show.progress = true;
    show.stop = true;
    if (!count && state.worker.status !== 'ready') {
      title = 'Getting ready…';
      detail = 'Loading the inspection models. This takes a few seconds.';
    } else {
      title = `Inspecting… ${count} of ${state.total || '?'} trays`;
      detail = 'Keep this window open. Each result is saved as soon as it is done.';
    }
  } else if (state.phase === 'done') {
    tone = toCheck || skipped ? 'warn' : 'good';
    title = toCheck ? `✓ Done. ${plural(toCheck, 'tray')} ${toCheck === 1 ? 'needs' : 'need'} checking`
      : skipped ? `✓ Done. ${plural(skipped, 'tray')} skipped` : '✓ Done. All trays OK';
    detail = `${plural(count, 'tray')} inspected. All results are saved.`;
    show.check = toCheck > 0;
    show.output = true;
  } else if (state.phase === 'stopped') {
    tone = 'warn';
    title = 'Stopped';
    detail = `${count} of ${state.total || count} trays were inspected before stopping. Those results are saved.`;
    show.output = count > 0;
  } else if (state.phase === 'failed') {
    tone = 'bad';
    title = 'The inspection could not finish';
    detail = state.error || 'Something went wrong. Press “Show details” and send the text to support.';
    show.details = true;
    show.output = count > 0;
  }

  $('banner').className = `banner ${tone}`;
  $('banner-title').textContent = title;
  $('banner-detail').textContent = detail;
  $('banner-progress').hidden = !show.progress;
  const percent = state.total ? Math.round((100 * count) / state.total) : 0;
  $('progress-bar').style.width = `${percent}%`;
  $('progress').setAttribute('aria-valuenow', String(percent));
  $('stop').hidden = !show.stop;
  $('show-check').hidden = !show.check;
  $('open-output').hidden = !show.output;
  $('open-output').title = state.outputDir;
  $('show-details').hidden = !show.details;
}

// ---------- Viewer ----------

function toggleSlot(slot) {
  // "Not sure" becomes lens first; Cancel restores the model's labels.
  const next = effectiveLabel(slot) === 'TLens' ? 'Empty' : 'TLens';
  if (next === slot.label) state.pending.delete(slotKey(slot));
  else state.pending.set(slotKey(slot), next);
  renderViewer();
  $('overlay').querySelector(`[data-key="${slotKey(slot)}"]`)?.focus();
}

function renderOverlay() {
  const svg = $('overlay');
  const image = $('viewer-img');
  const result = state.selected;
  svg.replaceChildren();
  // Outside edit mode the overlay only highlights "not sure" slots, which are hard to spot on a full tray.
  const unsure = result?.slots?.filter((slot) => effectiveLabel(slot) === 'Uncertain') ?? [];
  const visible = Boolean(result?.imageUrl) && (state.editing || unsure.length > 0);
  svg.toggleAttribute('hidden', !visible); // SVG elements have no .hidden property
  svg.classList.toggle('passive', !state.editing);
  if (!visible) return;
  if (!image.complete || !image.naturalWidth) {
    image.addEventListener('load', renderOverlay, { once: true });
    return;
  }
  svg.setAttribute('viewBox', `0 0 ${image.naturalWidth} ${image.naturalHeight}`);
  const { x: width, y: height } = result.pitch;
  const stroke = Math.min(width, height) * 0.08;
  if (!state.editing) {
    for (const slot of unsure) {
      svg.append(svgElement('rect', {
        class: 'unsure', x: slot.x - width * 0.6, y: slot.y - height * 0.6, width: width * 1.2, height: height * 1.2, 'stroke-width': stroke * 2,
      }));
    }
    return;
  }
  for (const slot of result.slots) {
    const label = effectiveLabel(slot);
    const changed = state.pending.has(slotKey(slot));
    const spoken = { TLens: 'lens', Empty: 'empty', Uncertain: 'not sure' }[label];
    const group = svgElement('g', {
      class: `slot${changed ? ' changed' : ''}${label === 'Uncertain' ? ' unsure' : ''}`, tabindex: 0, role: 'button', 'data-key': slotKey(slot),
      'aria-label': `Row ${slot.row}, column ${slot.column}: ${spoken}${changed ? ', changed' : ''}`,
    });
    group.append(svgElement('rect', { x: slot.x - width / 2, y: slot.y - height / 2, width, height, 'stroke-width': stroke }));
    if (changed && label === 'TLens') {
      group.append(svgElement('circle', { cx: slot.x, cy: slot.y, r: Math.min(width, height) * 0.22, 'stroke-width': stroke }));
    } else if (changed) {
      const dx = width * 0.25;
      const dy = height * 0.25;
      group.append(svgElement('path', {
        d: `M${slot.x - dx} ${slot.y - dy}L${slot.x + dx} ${slot.y + dy}M${slot.x + dx} ${slot.y - dy}L${slot.x - dx} ${slot.y + dy}`,
        'stroke-width': stroke,
      }));
    }
    group.addEventListener('click', () => toggleSlot(slot));
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggleSlot(slot); }
    });
    svg.append(group);
  }
}

// Scale the image to fit the centre area; the overlay follows because it covers the same box.
function fitImage() {
  const image = $('viewer-img');
  const area = $('viewer-area');
  if (!image.naturalWidth || $('stage').hidden) return;
  const style = getComputedStyle(area);
  const width = area.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const height = area.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
  const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  image.style.width = `${Math.floor(image.naturalWidth * scale)}px`;
  image.style.height = `${Math.floor(image.naturalHeight * scale)}px`;
}
new ResizeObserver(fitImage).observe($('viewer-area'));
$('viewer-img').addEventListener('load', fitImage);

// Plain-language verdict: what the operator should do with this tray.
function renderVerdict(result) {
  const box = $('verdict');
  const lines = [];
  let tone = 'good';
  if (result.status === 'skipped') {
    tone = 'bad';
    lines.push('Skipped: the app could not tell if this is a bare or a packaged tray.',
      'Open Settings, choose the tray type, and inspect again.');
  } else if (result.status === 'complete') {
    lines.push('✓ OK. No problems found.');
  } else if (result.status === 'reviewed') {
    lines.push(`✓ Checked and saved by you${result.corrected ? ` (${plural(result.corrected, 'slot')} fixed)` : ''}.`);
  } else {
    tone = 'warn';
    lines.push('Needs checking:');
    if (result.uncertain) lines.push(`• ${plural(result.uncertain, 'slot')} marked yellow (not sure). Click “Fix slots” and set each one to lens or empty.`);
    if (result.blurred) lines.push('• The photo looks blurry. Check the marks carefully, or retake the photo.');
    if (result.qr !== 'decoded') lines.push('• The QR code could not be read, so the lot number is unknown.');
    lines.push('When everything looks right, click “Fix slots”, then “Mark as checked”.');
  }
  box.className = `verdict ${tone}`;
  box.replaceChildren(...lines.map((text, index) => {
    const line = document.createElement(index === 0 ? 'strong' : 'span');
    line.textContent = text;
    return line;
  }));
  box.hidden = false;
}

function renderViewer() {
  const result = state.selected;
  const shown = Boolean(result && result.imageUrl);
  $('stage').hidden = !shown;
  $('viewer-foot').hidden = !shown;
  $('enhance-label').hidden = !shown;
  $('viewer-placeholder').hidden = shown;
  if (!result) {
    $('verdict').hidden = true;
    $('viewer-name').textContent = 'No tray selected';
    $('viewer-meta').textContent = 'Results appear here while the inspection runs.';
    return;
  }
  renderVerdict(result);
  if (!shown) {
    $('viewer-name').textContent = result.name;
    $('viewer-meta').textContent = 'Not inspected';
    return;
  }
  $('enhance').checked = state.enhance[result.tray];
  $('viewer-img').classList.toggle('enhanced', state.enhance[result.tray]);
  $('viewer-name').textContent = result.name.replace(/\.[^.]+$/, '');
  $('viewer-meta').textContent = `${TRAY_NAMES[result.tray]} tray · Lot ${result.lot ?? 'unknown'}`;
  if ($('viewer-img').getAttribute('src') !== result.imageUrl) $('viewer-img').src = result.imageUrl;
  $('viewer-img').alt = `${TRAY_NAMES[result.tray]} tray ${result.name} with lens and empty marks`;
  const labels = result.slots.map(effectiveLabel);
  $('viewer-tlens').textContent = labels.filter((label) => label === 'TLens').length;
  $('viewer-empty').textContent = labels.filter((label) => label === 'Empty').length;
  $('viewer-uncertain').textContent = labels.filter((label) => label === 'Uncertain').length;
  $('edit-bar').hidden = !state.editing;
  $('viewer-actions').hidden = state.editing;
  $('edit').disabled = state.phase === 'running';
  $('edit').title = state.phase === 'running' ? 'Wait until the inspection is done' : '';
  const unsure = labels.filter((label) => label === 'Uncertain').length;
  $('edit-count').textContent = [
    state.pending.size ? `${plural(state.pending.size, 'change')} not saved yet.` : '',
    unsure ? `${plural(unsure, 'yellow slot')} left: click ${unsure === 1 ? 'it' : 'each one'} to set lens or empty.` : '',
  ].filter(Boolean).join(' ') || 'No changes yet.';
  $('save').textContent = state.pending.size ? `Save ${plural(state.pending.size, 'change')}` : 'Mark as checked';
  // Nothing to save and yellow slots left: marking as checked would not clear the tray, so block it.
  $('save').disabled = !state.pending.size && unsure > 0;
  renderOverlay();
}

// ---------- Lists and summary ----------

function render() {
  const { results } = state;
  const toCheck = byStatus('review_required');
  const skipped = byStatus('skipped');
  $('summary-card').hidden = !results.length && state.phase === 'idle';
  $('stat-ok').textContent = byStatus('complete').length;
  $('stat-check').textContent = toCheck.length;
  $('stat-checked').textContent = byStatus('reviewed').length;
  $('stat-skipped').textContent = skipped.length;
  const bare = results.filter((r) => r.tray === 'bare').length;
  const packaged = results.filter((r) => r.tray === 'packaged').length;
  $('stat-types').textContent = results.length ? `${bare} bare · ${packaged} packaged` : '';
  $('count-all').textContent = results.length;
  $('count-check').textContent = toCheck.length;
  $('count-skipped').textContent = skipped.length;

  const shown = state.filter === 'check' ? toCheck : state.filter === 'skipped' ? skipped : results;
  const rows = shown.map((result) => {
    const tr = document.createElement('tr');
    tr.tabIndex = 0;
    tr.setAttribute('aria-selected', String(result === state.selected));
    tr.append(
      cell(result.name),
      result.tray ? chipCell(TRAY_NAMES[result.tray], result.tray) : cell('–'),
      cell(result.tlens, 'num'), cell(result.empty, 'num'),
      chipCell(STATUS_LABELS[result.status] ?? result.status, result.status),
    );
    tr.addEventListener('click', () => select(result, true));
    tr.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(result, true); } });
    return tr;
  });
  $('rows').replaceChildren(...rows);
  $('empty-table').hidden = rows.length > 0;
  $('empty-table').textContent = results.length ? 'No trays in this list.' : 'No trays inspected yet.';
  renderBanner();
  renderViewer();
}

function setFilter(filter) {
  state.filter = filter;
  pressOne($('filters'), $('filters').querySelector(`[data-filter="${filter}"]`));
  render();
}

function setControlsDisabled(disabled) {
  document.querySelectorAll('#pick-folder, #pick-image, #rerun, .settings input, .settings button')
    .forEach((element) => { element.disabled = disabled; });
}

// ---------- Running ----------

async function startRun() {
  if (!state.source || state.phase === 'running') return;
  if (!confirmDiscard()) return;
  const outputDir = $('output').value.trim();
  stopEditing();
  Object.assign(state, {
    results: [], total: 0, selected: null, userPicked: false, cancelled: false, error: '', phase: 'running', outputDir,
  });
  setFilter('all');
  $('log').replaceChildren();
  $('log-box').open = false;
  setControlsDisabled(true);
  render();
  try {
    await api.run({ input: state.source.path, tray: state.tray, outputDir, confidence: Number($('confidence').value), cpuThreads: Number($('threads').value) });
  } catch (error) {
    state.phase = 'failed';
    state.error = cleanError(error);
    appendLog(state.error);
    setControlsDisabled(false);
    render();
  }
}

async function pickAndRun(kind) {
  if (state.phase === 'running') return;
  const picked = await api.pick(kind);
  if (!picked) return;
  state.source = { kind, path: picked };
  $('source').hidden = false;
  $('source-kind').textContent = kind === 'image' ? 'Selected photo' : 'Selected folder';
  $('source-name').textContent = baseName(picked);
  $('source-path').textContent = picked;
  $('source-path').title = picked;
  startRun();
}

$('pick-folder').addEventListener('click', () => pickAndRun('folder'));
$('pick-image').addEventListener('click', () => pickAndRun('image'));
$('rerun').addEventListener('click', startRun);
$('pick-output').addEventListener('click', async () => {
  const picked = await api.pick('folder');
  if (picked) $('output').value = picked;
});

$('tray').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  state.tray = button.dataset.tray;
  pressOne($('tray'), button);
});

$('filters').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (button) setFilter(button.dataset.filter);
});

$('stop').addEventListener('click', () => {
  state.cancelled = true;
  api.cancel();
});
$('show-check').addEventListener('click', () => {
  setFilter('check');
  const first = byStatus('review_required')[0];
  if (first) select(first, true);
});
$('open-output').addEventListener('click', () => api.open(state.outputDir));
$('show-details').addEventListener('click', () => { $('log-box').open = true; $('log-box').scrollIntoView(); });

// ---------- Viewer actions ----------

$('enhance').addEventListener('change', () => {
  state.enhance[state.selected.tray] = $('enhance').checked;
  renderViewer();
});

$('edit').addEventListener('click', () => {
  state.editing = true;
  state.pending.clear();
  renderViewer();
});

$('discard').addEventListener('click', () => {
  stopEditing();
  renderViewer();
});

$('save').addEventListener('click', async () => {
  const changes = [...state.pending].map(([key, label]) => {
    const [row, column] = key.split(',').map(Number);
    return { row, column, label };
  });
  $('save').disabled = true;
  $('discard').disabled = true;
  try {
    const updated = await api.correct(state.selected.jsonPath, changes);
    state.results[state.results.indexOf(state.selected)] = updated;
    state.selected = updated;
    stopEditing();
    render();
    toast(updated.status === 'reviewed' ? '✓ Saved. Tray marked as checked.'
      : '✓ Saved. Some slots are still “not sure”, so the tray still needs checking.');
  } catch (error) {
    appendLog(cleanError(error));
    toast('Could not save. Press “Technical details” at the bottom for more information.');
    $('log-box').open = true;
  } finally {
    $('save').disabled = false;
    $('discard').disabled = false;
  }
});

$('open-folder').addEventListener('click', () => state.selected && api.open(state.selected.folder));

// ---------- Events from the main process ----------

api.on('log', appendLog);
api.on('total', (total) => { state.total = total; render(); });
api.on('result', (result) => {
  state.results.push(result);
  // Follow the newest tray until the operator picks one.
  if (!state.userPicked && result.imageUrl) state.selected = result;
  render();
});
api.on('done', ({ code }) => {
  setControlsDisabled(false);
  const skipped = byStatus('skipped').length;
  if (state.cancelled) state.phase = 'stopped';
  else if (code === 0 || (skipped && state.results.length)) state.phase = 'done';
  else state.phase = 'failed';
  if (state.phase === 'failed') $('log-box').open = true;
  // Show the first tray that needs attention, unless the operator already picked one.
  const first = byStatus('review_required')[0] ?? byStatus('skipped')[0];
  if (state.phase === 'done' && first && !state.userPicked) state.selected = first;
  render();
});

const envProblems = [];

function renderEnv(worker) {
  state.worker = worker;
  const workerText = {
    loading: ['● Getting ready…', 'loading'],
    ready: ['● Ready', 'ok'],
    stopped: ['● Restarts when you inspect again', 'bad'],
  }[worker.status];
  const [text, tone] = envProblems.length ? [`● ${envProblems.join(' · ')}`, 'bad'] : workerText;
  $('env').textContent = text;
  $('env').className = `env ${tone}`;
  renderBanner();
}

api.on('worker', renderEnv);

api.status().then((status) => {
  $('output').value = status.defaultOutput;
  if (!status.python) envProblems.push('Not installed correctly: Python environment missing (run “uv sync”)');
  for (const [tray, ok] of Object.entries(status.models)) if (!ok) envProblems.push(`${tray} model missing`);
  renderEnv(status.worker);
  setControlsDisabled(!status.python);
});

render();
