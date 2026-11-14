// Tray Inspector desktop shell: runs the frozen inspection worker (src/worker.py) and relays its progress to the
// window. A port of the former Electron main process; renderer.js reaches these commands through ui/bridge.js.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Mutex};
use std::thread;

use serde::Deserialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, RunEvent};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;

// Same folders as MODEL_DIRS in run.py; only used to warn early.
const MODELS: [(&str, &str); 2] = [("bare", "models/tlens-mobilenet-bare"), ("packaged", "models/tlens-mobilenet")];
const IMAGE_EXTENSIONS: [&str; 7] = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"];

struct Job {
    run: bool, // run jobs report results; correction jobs only log
    done: mpsc::Sender<i32>,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    ready: bool,
    pending: Option<String>, // job line to send once the models have loaded
    job: Option<Job>,
    restart: bool,
}

#[derive(Default)]
struct Inner {
    worker: Option<Worker>,
    generation: u64, // output from a worker that has since been replaced is ignored
    status: Value,
    busy: bool,                 // one job at a time: a run or a correction save
    openable: HashSet<PathBuf>, // only paths that came from run results may be opened
}

struct AppState {
    root: PathBuf, // installed resources: worker/ and models/
    inner: Mutex<Inner>,
}

fn log(app: &AppHandle, line: &str) {
    let _ = app.emit("log", line);
}

fn worker_exe(root: &Path) -> PathBuf {
    root.join("worker").join("tray-worker.exe")
}

fn set_status(app: &AppHandle, inner: &mut Inner, status: &str, detail: Option<String>) {
    inner.status = json!({ "status": status, "detail": detail });
    let _ = app.emit("worker", inner.status.clone());
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

// Slot spacing in image pixels, from neighbouring classification centres.
fn slot_pitch(slots: &[Value]) -> Value {
    let centres: HashMap<(i64, i64), (f64, f64)> = slots
        .iter()
        .filter_map(|s| Some(((s["row"].as_i64()?, s["column"].as_i64()?), (s["crop_center_x"].as_f64()?, s["crop_center_y"].as_f64()?))))
        .collect();
    let gaps = |d_row: i64, d_column: i64| -> Vec<f64> {
        centres
            .iter()
            .filter_map(|(&(row, column), &(x, y))| centres.get(&(row + d_row, column + d_column)).map(|&(nx, ny)| (nx - x).hypot(ny - y)))
            .collect()
    };
    json!({ "x": median(gaps(0, 1)), "y": median(gaps(1, 0)) })
}

fn read_result(root: &Path, inner: &mut Inner, json_path: &str) -> Result<Value, String> {
    let resolved = root.join(json_path);
    let text = std::fs::read_to_string(&resolved).map_err(|e| e.to_string())?;
    let result: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let folder = resolved.parent().map(Path::to_path_buf).unwrap_or_default();
    inner.openable.insert(resolved.clone());
    inner.openable.insert(folder.clone());
    let all_slots = result["slots"].as_array().cloned().unwrap_or_default();
    let slots: Vec<Value> = all_slots
        .iter()
        .filter(|s| s["prediction"] != "Excluded")
        .map(|s| json!({
            "row": s["row"], "column": s["column"], "label": s["prediction"],
            "model": s.get("model_prediction").cloned().unwrap_or(Value::Null),
            "x": s["center_x"], "y": s["center_y"],
        }))
        .collect();
    let version = std::fs::metadata(&resolved)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis());
    let image = result["image"].as_str().unwrap_or_default();
    Ok(json!({
        "slots": slots,
        "pitch": slot_pitch(&all_slots),
        "corrected": result.get("corrected_count").and_then(Value::as_i64).unwrap_or(0),
        "blurred": result["image_quality"]["is_severely_blurred"].as_bool().unwrap_or(false),
        "name": Path::new(image).file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
        "tray": result["tray_type"],
        "traySource": result["tray_type_source"],
        "lot": result["lot_number"],
        "qr": result["qr_status"],
        "tlens": result["occupied_count"],
        "empty": result["empty_count"],
        "uncertain": result["uncertain_count"],
        "status": result["inspection_status"],
        "jsonPath": resolved,
        "folder": folder,
        "imagePath": root.join(result["annotated_image"].as_str().unwrap_or_default()),
        "version": version,
    }))
}

fn handle_run_line(app: &AppHandle, root: &Path, inner: &mut Inner, line: &str) {
    log(app, line);
    if let Some(total) = line.strip_prefix("Found ").and_then(|rest| rest.split_once(" input images")).and_then(|(n, _)| n.parse::<u64>().ok()) {
        let _ = app.emit("total", total);
    } else if let Some(index) = line.find(" json=") {
        let path = &line[index + " json=".len()..];
        match read_result(root, inner, path) {
            Ok(result) => {
                let _ = app.emit("result", result);
            }
            Err(error) => log(app, &format!("Could not read result {path}: {error}")),
        }
    } else if let Some(index) = line.rfind(": SKIPPED") {
        let _ = app.emit("result", json!({ "name": &line[..index], "status": "skipped" }));
    }
}

fn start_worker(app: &AppHandle, root: &Path, inner: &mut Inner) {
    inner.generation += 1;
    let generation = inner.generation;
    let mut command = Command::new(worker_exe(root));
    command.current_dir(root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return set_status(app, inner, "stopped", Some(format!("could not start the worker: {error}"))),
    };
    let (stdin, stdout, stderr) = (child.stdin.take().unwrap(), child.stdout.take().unwrap(), child.stderr.take().unwrap());
    inner.worker = Some(Worker { child, stdin, ready: false, pending: None, job: None, restart: false });
    set_status(app, inner, "loading", None);
    let handle = app.clone();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            log(&handle, &line);
        }
    });
    let handle = app.clone();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            on_line(&handle, generation, &line);
        }
        on_exit(&handle, generation);
    });
}

fn on_line(app: &AppHandle, generation: u64, line: &str) {
    let state = app.state::<AppState>();
    let mut inner = state.inner.lock().unwrap();
    if inner.generation != generation {
        return;
    }
    let Some(worker) = inner.worker.as_mut() else { return };
    if line == "@@ready" {
        worker.ready = true;
        if let Some(job_line) = worker.pending.take() {
            let _ = worker.stdin.write_all(job_line.as_bytes()).and_then(|_| worker.stdin.flush());
        }
        return set_status(app, &mut inner, "ready", None);
    }
    if let Some(code) = line.strip_prefix("@@done ").and_then(|code| code.trim().parse::<i32>().ok()) {
        if let Some(job) = worker.job.take() {
            let _ = job.done.send(code);
            return;
        }
    }
    match worker.job.as_ref().map(|job| job.run) {
        Some(true) => handle_run_line(app, &state.root, &mut inner, line),
        _ => log(app, line),
    }
}

fn on_exit(app: &AppHandle, generation: u64) {
    let state = app.state::<AppState>();
    let mut inner = state.inner.lock().unwrap();
    if inner.generation != generation {
        return;
    }
    let Some(mut worker) = inner.worker.take() else { return };
    if let Some(job) = worker.job.take() {
        let _ = job.done.send(-1);
    }
    let exit = worker.child.wait().map(|s| s.to_string()).unwrap_or_else(|e| e.to_string());
    if worker.restart {
        start_worker(app, &state.root, &mut inner);
    } else {
        set_status(app, &mut inner, "stopped", Some(format!("exited ({exit})")));
    }
}

// Resolves with the job's exit code; -1 if the worker stopped (cancel or crash).
fn submit_job(app: &AppHandle, root: &Path, inner: &mut Inner, command: &str, argv: Vec<String>, run: bool) -> mpsc::Receiver<i32> {
    if inner.worker.is_none() {
        start_worker(app, root, inner);
    }
    let (done, receiver) = mpsc::channel();
    let line = format!("{}\n", json!({ "command": command, "argv": argv }));
    match inner.worker.as_mut() {
        None => {
            let _ = done.send(-1);
        }
        Some(worker) => {
            worker.job = Some(Job { run, done });
            if worker.ready {
                let _ = worker.stdin.write_all(line.as_bytes()).and_then(|_| worker.stdin.flush());
            } else {
                log(app, "Waiting for the models to finish loading…");
                worker.pending = Some(line);
            }
        }
    }
    receiver
}

#[tauri::command]
fn status(app: AppHandle) -> Value {
    let state = app.state::<AppState>();
    let inner = state.inner.lock().unwrap();
    let models: serde_json::Map<String, Value> =
        MODELS.iter().map(|(tray, dir)| (tray.to_string(), json!(state.root.join(dir).is_dir()))).collect();
    let documents = app.path().document_dir().unwrap_or_else(|_| std::env::temp_dir());
    json!({
        "python": worker_exe(&state.root).is_file(),
        "models": models,
        "defaultOutput": documents.join("Tray Inspector"),
        "worker": inner.status.clone(),
    })
}

#[tauri::command]
async fn pick(app: AppHandle, kind: String) -> Option<PathBuf> {
    let dialog = app.dialog().file();
    let picked = if kind == "image" {
        dialog.add_filter("Images", &IMAGE_EXTENSIONS).blocking_pick_file()
    } else {
        dialog.blocking_pick_folder()
    };
    picked.and_then(|path| path.into_path().ok())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunOptions {
    input: String,
    tray: String,
    output_dir: String,
    confidence: f64,
    cpu_threads: u32,
}

#[tauri::command]
fn run(app: AppHandle, options: RunOptions) -> Result<(), String> {
    let state = app.state::<AppState>();
    let mut inner = state.inner.lock().unwrap();
    if inner.busy {
        return Err("A run is already in progress.".into());
    }
    if !["auto", "bare", "packaged"].contains(&options.tray.as_str()) {
        return Err(format!("Unknown tray type: {}", options.tray));
    }
    let input = Path::new(&options.input);
    if !input.exists() {
        return Err(format!("Input does not exist: {}", options.input));
    }
    if input.is_file() {
        let _ = app.emit("total", 1);
    }
    // Same arguments as `python run.py ...`; passed as a list, so paths need no quoting.
    let args = vec![
        options.tray.clone(), options.input.clone(),
        "--output-dir".into(), options.output_dir.clone(),
        "--confidence".into(), options.confidence.to_string(),
        "--cpu-threads".into(), options.cpu_threads.to_string(),
    ];
    log(&app, &format!("> run.py {}", args.join(" ")));
    inner.openable.insert(state.root.join(&options.output_dir));
    inner.busy = true;
    let receiver = submit_job(&app, &state.root, &mut inner, "run", args, true);
    let handle = app.clone();
    thread::spawn(move || {
        let code = receiver.recv().unwrap_or(-1);
        handle.state::<AppState>().inner.lock().unwrap().busy = false;
        let _ = handle.emit("done", json!({ "code": code }));
    });
    Ok(())
}

#[derive(Deserialize)]
struct Change {
    row: i64,
    column: i64,
    label: String,
}

#[tauri::command]
async fn correct(app: AppHandle, json_path: String, changes: Vec<Change>) -> Result<Value, String> {
    let receiver = {
        let state = app.state::<AppState>();
        let mut inner = state.inner.lock().unwrap();
        if !inner.openable.contains(Path::new(&json_path)) {
            return Err("Unknown result file.".into());
        }
        if inner.busy {
            return Err("Wait for the current run to finish before saving corrections.".into());
        }
        let mut args = vec![json_path.clone()];
        for change in &changes {
            if !["TLens", "Empty"].contains(&change.label.as_str()) {
                return Err(format!("Invalid correction label: {}", change.label));
            }
            args.push("--set".into());
            args.push(format!("{},{}={}", change.row, change.column, change.label));
        }
        inner.busy = true;
        submit_job(&app, &state.root, &mut inner, "corrections", args, false)
    };
    let code = tauri::async_runtime::spawn_blocking(move || receiver.recv().unwrap_or(-1)).await.unwrap_or(-1);
    let state = app.state::<AppState>();
    let mut inner = state.inner.lock().unwrap();
    inner.busy = false;
    if code != 0 {
        return Err(format!("Saving corrections failed (exit {code}); see log."));
    }
    read_result(&state.root, &mut inner, &json_path)
}

// Stopping mid-run means stopping the worker; a fresh one starts loading straight away.
#[tauri::command]
fn cancel(app: AppHandle) {
    let state = app.state::<AppState>();
    let mut inner = state.inner.lock().unwrap();
    if inner.busy {
        if let Some(worker) = inner.worker.as_mut() {
            worker.restart = true;
            let _ = worker.child.kill();
        }
    }
}

#[tauri::command]
fn open(app: AppHandle, target: String) -> Result<(), String> {
    let allowed = app.state::<AppState>().inner.lock().unwrap().openable.contains(Path::new(&target));
    if !allowed {
        return Ok(());
    }
    app.opener().open_path(target, None::<&str>).map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let root = app.path().resource_dir()?;
            let status = json!({ "status": "stopped", "detail": "not started" });
            app.manage(AppState { root: root.clone(), inner: Mutex::new(Inner { status, ..Default::default() }) });
            if worker_exe(&root).is_file() {
                // Load the models while the user picks an input.
                let state = app.state::<AppState>();
                start_worker(app.handle(), &root, &mut state.inner.lock().unwrap());
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![status, pick, run, correct, cancel, open])
        .build(tauri::generate_context!())
        .expect("error while building Tray Inspector")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(worker) = app.state::<AppState>().inner.lock().unwrap().worker.as_mut() {
                    let _ = worker.child.kill();
                }
            }
        });
}
