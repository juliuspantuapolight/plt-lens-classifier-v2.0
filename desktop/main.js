// Electron main process: runs the repo's run.py and relays its progress to the window.
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const REPO = path.resolve(__dirname, '..');
const PYTHON = path.join(REPO, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
// Same folders as MODEL_DIRS in run.py; only used to warn early.
const MODELS = { bare: 'models/tlens-mobilenet-bare', packaged: 'models/tlens-mobilenet' };
const IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp'];

let win = null;
let busy = false; // one job at a time: a run or a correction save
const openable = new Set(); // only paths that came from run results may be opened

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
}

// Slot spacing in image pixels, from neighbouring classification centres.
function slotPitch(slots) {
  const byPosition = new Map(slots.map((slot) => [`${slot.row},${slot.column}`, slot]));
  const gaps = (dRow, dColumn) => slots.flatMap((slot) => {
    const next = byPosition.get(`${slot.row + dRow},${slot.column + dColumn}`);
    return next ? [Math.hypot(next.crop_center_x - slot.crop_center_x, next.crop_center_y - slot.crop_center_y)] : [];
  });
  return { x: median(gaps(0, 1)), y: median(gaps(1, 0)) };
}

function readResult(jsonPath) {
  const resolved = path.resolve(REPO, jsonPath);
  const result = JSON.parse(fs.readFileSync(resolved, 'utf8'));
  const folder = path.dirname(resolved);
  openable.add(resolved);
  openable.add(folder);
  return {
    slots: result.slots.filter((slot) => slot.prediction !== 'Excluded').map((slot) => ({
      row: slot.row, column: slot.column, label: slot.prediction, model: slot.model_prediction ?? null,
      x: slot.center_x, y: slot.center_y,
    })),
    pitch: slotPitch(result.slots),
    corrected: result.corrected_count ?? 0,
    blurred: Boolean(result.image_quality?.is_severely_blurred),
    name: path.basename(result.image),
    tray: result.tray_type,
    traySource: result.tray_type_source,
    lot: result.lot_number,
    qr: result.qr_status,
    tlens: result.occupied_count,
    empty: result.empty_count,
    uncertain: result.uncertain_count,
    status: result.inspection_status,
    jsonPath: resolved,
    folder,
    // The query string makes the window reload the image after it is redrawn.
    imageUrl: `${pathToFileURL(path.resolve(REPO, result.annotated_image)).href}?v=${fs.statSync(resolved).mtimeMs}`,
  };
}

function handleLine(line) {
  send('log', line);
  let match;
  if ((match = line.match(/^Found (\d+) input images/))) {
    send('total', Number(match[1]));
  } else if ((match = line.match(/ json=(.+)$/))) {
    try {
      send('result', readResult(match[1]));
    } catch (error) {
      send('log', `Could not read result ${match[1]}: ${error.message}`);
    }
  } else if ((match = line.match(/^(.+): SKIPPED/))) {
    send('result', { name: match[1], status: 'skipped' });
  }
}

function streamLines(stream, onLine) {
  let rest = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk) => {
    const lines = (rest + chunk).split(/\r?\n/);
    rest = lines.pop();
    lines.forEach(onLine);
  });
  stream.on('end', () => rest && onLine(rest));
}

// --- Background Python worker (src/worker.py): imports torch and warms the models once, at startup. ---
let worker = null; // { proc, ready, readyPromise, job, restart }
let workerStatus = { status: 'stopped', detail: 'not started' };

function setWorkerStatus(status, detail) {
  workerStatus = { status, detail };
  send('worker', workerStatus);
}

function startWorker() {
  const current = { job: null, ready: false, restart: false };
  let settleReady;
  current.readyPromise = new Promise((resolve) => { settleReady = resolve; });
  current.proc = spawn(PYTHON, ['-m', 'src.worker'], {
    cwd: REPO,
    windowsHide: true,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
  });
  worker = current;
  setWorkerStatus('loading');
  streamLines(current.proc.stdout, (line) => {
    if (line === '@@ready') {
      current.ready = true;
      settleReady(true);
      setWorkerStatus('ready');
      return;
    }
    const done = line.match(/^@@done (-?\d+)$/);
    if (done && current.job) {
      const { resolve } = current.job;
      current.job = null;
      resolve(Number(done[1]));
    } else if (current.job) {
      current.job.onLine(line);
    } else {
      send('log', line);
    }
  });
  streamLines(current.proc.stderr, (line) => send('log', line));
  let exited = false;
  const onExit = (detail) => {
    if (exited) return;
    exited = true;
    if (worker === current) worker = null;
    settleReady(false);
    if (current.job) current.job.resolve(-1);
    current.job = null;
    if (current.restart) startWorker();
    else setWorkerStatus('stopped', detail);
  };
  current.proc.on('error', (error) => onExit(`could not start Python: ${error.message}`));
  current.proc.on('close', (code) => onExit(`exited with code ${code}`));
}

// Resolves with the job's exit code; -1 if the worker stopped (cancel or crash).
async function runJob(command, argv, onLine) {
  if (!worker) startWorker();
  const current = worker;
  if (!current.ready) send('log', 'Waiting for the models to finish loading…');
  if (!(await current.readyPromise)) return -1;
  return new Promise((resolve) => {
    current.job = { onLine, resolve };
    current.proc.stdin.write(`${JSON.stringify({ command, argv })}\n`);
  });
}

ipcMain.handle('status', () => ({
  python: fs.existsSync(PYTHON),
  models: Object.fromEntries(Object.entries(MODELS).map(([tray, dir]) => [tray, fs.existsSync(path.join(REPO, dir))])),
  defaultOutput: path.join(REPO, 'outputs', 'mobilenet-inspection'),
  worker: workerStatus,
}));

ipcMain.handle('pick', async (_event, kind) => {
  const options = kind === 'image'
    ? { properties: ['openFile'], filters: [{ name: 'Images', extensions: IMAGE_EXTENSIONS }] }
    : { properties: ['openDirectory', 'createDirectory'] };
  const { canceled, filePaths } = await dialog.showOpenDialog(win, options);
  return canceled ? null : filePaths[0];
});

ipcMain.handle('run', (_event, { input, tray, outputDir, confidence, cpuThreads }) => {
  if (busy) throw new Error('A run is already in progress.');
  if (!['auto', 'bare', 'packaged'].includes(tray)) throw new Error(`Unknown tray type: ${tray}`);
  if (!fs.existsSync(input)) throw new Error(`Input does not exist: ${input}`);
  if (fs.statSync(input).isFile()) send('total', 1);
  // Same arguments as `python run.py ...`; passed as a list, so paths need no quoting.
  const args = [tray, input, '--output-dir', outputDir, '--confidence', String(confidence), '--cpu-threads', String(cpuThreads)];
  send('log', `> run.py ${args.join(' ')}`);
  openable.add(path.resolve(REPO, outputDir));
  busy = true;
  runJob('run', args, handleLine).then((code) => {
    busy = false;
    send('done', { code });
  });
});

ipcMain.handle('correct', async (_event, { jsonPath, changes }) => {
  if (!openable.has(jsonPath)) throw new Error('Unknown result file.');
  if (busy) throw new Error('Wait for the current run to finish before saving corrections.');
  const args = [jsonPath];
  for (const { row, column, label } of changes) {
    if (!Number.isInteger(row) || !Number.isInteger(column) || !['TLens', 'Empty'].includes(label)) {
      throw new Error(`Invalid correction: ${JSON.stringify({ row, column, label })}`);
    }
    args.push('--set', `${row},${column}=${label}`);
  }
  busy = true;
  try {
    const code = await runJob('corrections', args, (line) => send('log', line));
    if (code !== 0) throw new Error(`Saving corrections failed (exit ${code}); see log.`);
    return readResult(jsonPath);
  } finally {
    busy = false;
  }
});

// Stopping mid-run means stopping the worker; a fresh one starts loading straight away.
ipcMain.handle('cancel', () => {
  if (busy && worker) {
    worker.restart = true;
    worker.proc.kill();
  }
});

ipcMain.handle('open', (_event, target) => {
  if (openable.has(target)) return shell.openPath(target);
});

app.whenReady().then(() => {
  win = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 900,
    minHeight: 600,
    title: 'Tray Inspector',
    backgroundColor: '#EEF0F3',
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, sandbox: true, nodeIntegration: false },
  });
  win.maximize();
  win.loadFile(path.join(__dirname, 'index.html'));
  if (fs.existsSync(PYTHON)) startWorker(); // load models while the user picks an input
});

app.on('window-all-closed', () => {
  if (worker) worker.proc.kill();
  app.quit();
});
