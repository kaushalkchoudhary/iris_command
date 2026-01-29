use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write, BufWriter};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::fs::File;
use rusqlite::Connection;

use crate::helpers::{
    load_rtsp_config, mask_rtsp_credentials, save_rtsp_config, source_display_name, OverlayConfig,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct OverlayState {
    pub heatmap: bool,
    pub trails: bool,
    pub bboxes: bool,
}

impl From<OverlayConfig> for OverlayState {
    fn from(config: OverlayConfig) -> Self {
        OverlayState {
            heatmap: config.heatmap,
            trails: config.trails,
            bboxes: config.bboxes,
        }
    }
}

impl From<OverlayState> for OverlayConfig {
    fn from(state: OverlayState) -> Self {
        OverlayConfig {
            heatmap: state.heatmap,
            trails: state.trails,
            bboxes: state.bboxes,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OverlayUpdate {
    heatmap: Option<bool>,
    trails: Option<bool>,
    bboxes: Option<bool>,
}

#[derive(Debug, Serialize)]
struct SourceInfo {
    index: usize,
    name: String,
    url: String,
    active: bool,
}

#[derive(Debug, Serialize)]
struct SourcesResponse {
    sources: Vec<SourceInfo>,
    active_sources: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct SetActiveSourcesRequest {
    /// 1-based indexes of sources to activate
    #[serde(default)]
    indexes: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct SourceIndexRequest {
    /// 1-based index of source
    index: usize,
}

#[derive(Debug, Deserialize)]
struct ToggleSourceRequest {
    /// 1-based index of source to toggle
    index: usize,
    /// true to activate, false to deactivate
    active: bool,
}

#[derive(Debug, Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

pub type OverlayMap = Arc<Mutex<HashMap<String, OverlayState>>>;

/// Upload job status tracking
#[derive(Debug, Clone, Serialize)]
pub struct UploadJob {
    pub id: String,
    pub filename: String,
    pub status: String,        // "processing", "done", "error"
    pub output_file: Option<String>,
    pub error: Option<String>,
}

pub type SharedJobMap = Arc<Mutex<HashMap<String, UploadJob>>>;

/// Real-time metrics for a source (updated by processing threads)
#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct SourceMetrics {
    pub fps: f32,
    pub congestion_index: i32,
    pub traffic_density: i32,
    pub mobility_index: i32,
    pub stalled_pct: i32,
    pub slow_pct: i32,
    pub medium_pct: i32,
    pub fast_pct: i32,
    pub detection_count: usize,
}

pub type MetricsMap = Arc<Mutex<HashMap<String, SourceMetrics>>>;

/// Running source handle with stop signal
pub struct RunningSource {
    pub index: usize,
    pub name: String,
    pub url: String,
    pub stop_signal: Arc<AtomicBool>,
    pub handle: Option<JoinHandle<()>>,
}

/// Manages dynamically started/stopped source processors
pub struct SourceManager {
    pub running: HashMap<usize, RunningSource>,
    pub rtsp_links: Vec<String>,
}

impl SourceManager {
    pub fn new(rtsp_links: Vec<String>) -> Self {
        Self {
            running: HashMap::new(),
            rtsp_links,
        }
    }

    pub fn is_running(&self, index: usize) -> bool {
        self.running.contains_key(&index)
    }

    pub fn get_running_indexes(&self) -> Vec<usize> {
        let mut indexes: Vec<usize> = self.running.keys().copied().collect();
        indexes.sort_unstable();
        indexes
    }

    pub fn stop_source(&mut self, index: usize) -> bool {
        if let Some(mut source) = self.running.remove(&index) {
            info!("Stopping source {} ({})...", index, source.name);
            source.stop_signal.store(true, Ordering::SeqCst);
            if let Some(handle) = source.handle.take() {
                let _ = handle.join();
            }
            info!("Source {} stopped", index);
            true
        } else {
            false
        }
    }

    pub fn stop_all(&mut self) {
        let indexes: Vec<usize> = self.running.keys().copied().collect();
        for index in indexes {
            self.stop_source(index);
        }
    }
}

pub type SharedSourceManager = Arc<Mutex<SourceManager>>;

/// Callback type for starting a source processor
pub type StartSourceFn =
    Arc<dyn Fn(usize, String, Arc<AtomicBool>) -> JoinHandle<()> + Send + Sync>;

/// Callback type for starting an upload-only processor (no MediaMTX, returns output path)
pub type StartUploadFn =
    Arc<dyn Fn(String, Arc<AtomicBool>) -> JoinHandle<Option<String>> + Send + Sync>;

/// Shared state for the control server including config path for persistence
pub struct ControlServerState {
    pub overlays: OverlayMap,
    pub config_path: PathBuf,
    pub db_path: PathBuf,
    pub source_manager: SharedSourceManager,
    pub start_source_fn: Option<StartSourceFn>,
    pub start_upload_fn: Option<StartUploadFn>,
    pub metrics: MetricsMap,
    pub jobs: SharedJobMap,
}

pub type SharedControlState = Arc<Mutex<ControlServerState>>;

pub fn start_control_server(
    addr: &str,
    overlays: OverlayMap,
    config_path: PathBuf,
    db_path: PathBuf,
    source_manager: SharedSourceManager,
    start_source_fn: StartSourceFn,
    start_upload_fn: StartUploadFn,
    metrics: MetricsMap,
    jobs: SharedJobMap,
) -> thread::JoinHandle<()> {
    let addr = addr.to_string();
    let state = Arc::new(Mutex::new(ControlServerState {
        overlays,
        config_path,
        db_path,
        source_manager,
        start_source_fn: Some(start_source_fn),
        start_upload_fn: Some(start_upload_fn),
        metrics,
        jobs,
    }));

    thread::spawn(move || {
        let listener = TcpListener::bind(&addr).expect("bind control server");
        info!("Control server listening on {}", addr);
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let state = Arc::clone(&state);
            thread::spawn(move || {
                let _ = handle_request(&mut stream, state);
            });
        }
    })
}

fn handle_request(stream: &mut TcpStream, state: SharedControlState) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    let mut data = Vec::new();
    let mut buf = [0u8; 8192]; // Larger buffer for faster initial reading
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 { break; }
        data.extend_from_slice(&buf[..n]);
        // Search for end of headers
        if data.windows(4).any(|w| w == b"\r\n\r\n") || data.len() > 128 * 1024 {
            break;
        }
    }

    let Some(header_end) = data.windows(4).position(|w| w == b"\r\n\r\n") else {
        return respond(stream, 400, "text/plain", "Bad Request");
    };

    let header_bytes = &data[..header_end + 4];
    let headers = String::from_utf8_lossy(header_bytes);
    let mut lines = headers.lines();
    let Some(request_line) = lines.next() else {
        return respond(stream, 400, "text/plain", "Bad Request");
    };
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    let mut content_length = 0usize;
    let mut expect_continue = false;
    for line in lines {
        let lower = line.to_lowercase();
        if lower.starts_with("content-length:") {
            content_length = lower["content-length:".len()..].trim().parse::<usize>().unwrap_or(0);
        } else if lower.starts_with("expect:") && lower.contains("100-continue") {
            expect_continue = true;
        }
    }

    // Handle Expect: 100-continue to avoid browser stalls
    if expect_continue {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }

    // Special handling for /api/upload to stream the body
    if method == "POST" && path.starts_with("/api/upload") {
        let query_part = path.split('?').nth(1).unwrap_or("");
        let mut filename = "upload.mp4".to_string();
        for pair in query_part.split('&') {
            if let Some(rest) = pair.strip_prefix("filename=") {
                filename = url_decode(rest);
            }
        }

        // Validate extension
        if !filename.ends_with(".mp4") && !filename.ends_with(".mkv") {
             return respond(stream, 400, "text/plain", "Only .mp4 and .mkv allowed");
        }

        let ctrl_state = state.lock().unwrap();
        let config_path = ctrl_state.config_path.clone();
        let start_upload_fn = ctrl_state.start_upload_fn.clone();
        let jobs = ctrl_state.jobs.clone();
        drop(ctrl_state);

        // Ensure upload dir exists
        let upload_dir = config_path.parent().unwrap().join("uploads").join("recordings");
        std::fs::create_dir_all(&upload_dir).unwrap_or(());
        let target_path = upload_dir.join(&filename);

        info!("Receiving upload: {} -> {}", filename, target_path.display());
        let mut file = match File::create(&target_path) {
            Ok(f) => f,
            Err(e) => {
                warn!("Failed to create upload file: {}", e);
                return respond(stream, 500, "text/plain", "File creation failed");
            }
        };

        // Write the part of body already read in header buffer
        if header_end + 4 < data.len() {
             file.write_all(&data[header_end + 4..])?;
        }
        let mut bytes_written = (data.len() - (header_end + 4)) as usize;

        // Optimized stream with larger buffer and longer timeout
        stream.set_read_timeout(Some(Duration::from_secs(60)))?;
        let mut upload_buf = vec![0u8; 65536];
        let mut writer = BufWriter::with_capacity(128 * 1024, file);

        while bytes_written < content_length {
            let remaining = content_length - bytes_written;
            let to_read = std::cmp::min(remaining, upload_buf.len());
            let n = stream.read(&mut upload_buf[..to_read])?;
            if n == 0 { break; }
            writer.write_all(&upload_buf[..n])?;
            bytes_written += n;
        }
        writer.flush()?;
        info!("Upload complete: {} bytes", bytes_written);
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;

        // Generate a job ID and spawn inference (no MediaMTX, no RTSP config)
        let job_id = format!("{}_{}", filename, chrono::Local::now().format("%Y%m%d_%H%M%S"));
        let full_path = target_path.to_string_lossy().to_string();

        {
            let mut job_map = jobs.lock().unwrap();
            job_map.insert(job_id.clone(), UploadJob {
                id: job_id.clone(),
                filename: filename.clone(),
                status: "processing".to_string(),
                output_file: None,
                error: None,
            });
        }

        if let Some(ref start_fn) = start_upload_fn {
            let job_id_clone = job_id.clone();
            let jobs_clone = jobs.clone();
            let handle = start_fn(full_path, Arc::new(AtomicBool::new(false)));

            // Spawn a watcher thread that waits for the inference to finish
            thread::spawn(move || {
                let result = handle.join();
                let mut job_map = jobs_clone.lock().unwrap();
                if let Some(job) = job_map.get_mut(&job_id_clone) {
                    match result {
                        Ok(Some(out_file)) => {
                            info!("Upload job {} completed: {}", job_id_clone, out_file);
                            job.status = "done".to_string();
                            job.output_file = Some(out_file);
                        }
                        Ok(None) => {
                            warn!("Upload job {} finished with no output", job_id_clone);
                            job.status = "error".to_string();
                            job.error = Some("Processing finished with no output file".to_string());
                        }
                        Err(_) => {
                            warn!("Upload job {} thread panicked", job_id_clone);
                            job.status = "error".to_string();
                            job.error = Some("Processing thread panicked".to_string());
                        }
                    }
                }
            });
        }

         let payload = serde_json::json!({
             "success": true,
             "job_id": job_id,
             "filename": filename
         });
         return respond(stream, 200, "application/json", &payload.to_string());
    }

    let mut body = data[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }

    // Handle CORS preflight
    if method == "OPTIONS" {
        let response = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nAccess-Control-Max-Age: 86400\r\n\r\n";
        stream.write_all(response.as_bytes())?;
        return Ok(());
    }

    debug!("Control API: {} {}", method, path);

    let ctrl_state = state.lock().unwrap();
    let overlays = ctrl_state.overlays.clone();
    let config_path = ctrl_state.config_path.clone();
    let db_path = ctrl_state.db_path.clone();
    let source_manager = ctrl_state.source_manager.clone();
    let start_source_fn = ctrl_state.start_source_fn.clone();
    let metrics_map = ctrl_state.metrics.clone();
    let jobs_map = ctrl_state.jobs.clone();
    drop(ctrl_state);

    if method == "GET" && path == "/api/overlays" {
        let map = overlays.lock().unwrap();
        debug!("Available streams: {:?}", map.keys().collect::<Vec<_>>());
        let payload = serde_json::to_string(&*map).unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    if let Some(stream_name) = path.strip_prefix("/api/overlays/") {
        let stream_name = stream_name.trim_matches('/');
        if stream_name.is_empty() {
            warn!("Control API: Empty stream name in request");
            return respond(stream, 400, "text/plain", "Bad Request");
        }

        if method == "GET" {
            let map = overlays.lock().unwrap();
            if let Some(state) = map.get(stream_name) {
                debug!("GET overlays for '{}': {:?}", stream_name, state);
                let payload = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
                return respond(stream, 200, "application/json", &payload);
            }
            warn!(
                "Control API: Stream '{}' not found. Available: {:?}",
                stream_name,
                map.keys().collect::<Vec<_>>()
            );
            return respond(stream, 404, "text/plain", "Not Found");
        }

        if method == "POST" {
            let update: OverlayUpdate = serde_json::from_slice(&body).unwrap_or(OverlayUpdate {
                heatmap: None,
                trails: None,
                bboxes: None,
            });
            debug!(
                "POST overlay update for '{}': heatmap={:?}, trails={:?}, bboxes={:?}",
                stream_name, update.heatmap, update.trails, update.bboxes
            );

            let next = {
                let mut map = overlays.lock().unwrap();

                // Check if stream exists
                if !map.contains_key(stream_name) {
                    warn!(
                        "Control API: Stream '{}' not found for update. Available: {:?}",
                        stream_name,
                        map.keys().collect::<Vec<_>>()
                    );
                }

                let current = map.get(stream_name).copied().unwrap_or_default();
                let next = OverlayState {
                    heatmap: update.heatmap.unwrap_or(current.heatmap),
                    trails: update.trails.unwrap_or(current.trails),
                    bboxes: update.bboxes.unwrap_or(current.bboxes),
                };
                map.insert(stream_name.to_string(), next);
                info!(
                    "[{}] Overlay state changed: heatmap={}, trails={}, bboxes={}",
                    stream_name, next.heatmap, next.trails, next.bboxes
                );
                next
            };

            // Persist overlay changes to config file
            if let Err(e) = persist_overlay_to_config(&config_path, stream_name, next) {
                warn!("Failed to persist overlay config: {}", e);
            }

            let payload = serde_json::to_string(&next).unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        }
    }

    // GET /api/sources - list all available RTSP sources with running status
    if method == "GET" && path == "/api/sources" {
        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };
        let mgr = source_manager.lock().unwrap();
        let running_indexes = mgr.get_running_indexes();
        drop(mgr);

        let sources: Vec<SourceInfo> = config
            .rtsp_links
            .iter()
            .enumerate()
            .map(|(i, url)| {
                let idx = i + 1;
                SourceInfo {
                    index: idx,
                    name: source_display_name(url),
                    url: mask_rtsp_credentials(url),
                    active: running_indexes.contains(&idx),
                }
            })
            .collect();
        let response = SourcesResponse {
            sources,
            active_sources: running_indexes,
        };
        let payload = serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // GET /api/sources - get all defined sources
    if method == "GET" && path == "/api/sources" {
        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };
        let payload = serde_json::to_string(&serde_json::json!({
            "sources": config.rtsp_links
        }))
        .unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // POST /api/sources/start - start processing a source (live, no restart needed)
    // Body: {"index": 3}
    if method == "POST" && path == "/api/sources/start" {
        let req: SourceIndexRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => {
                return respond(
                    stream,
                    400,
                    "text/plain",
                    "Invalid JSON: expected {\"index\": N}",
                )
            }
        };

        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };

        if req.index < 1 || req.index > config.rtsp_links.len() {
            return respond(
                stream,
                400,
                "text/plain",
                &format!(
                    "Invalid index {}: must be between 1 and {}",
                    req.index,
                    config.rtsp_links.len()
                ),
            );
        }

        let url = config.rtsp_links[req.index - 1].clone();
        let name = source_display_name(&url);

        let mut mgr = source_manager.lock().unwrap();
        if mgr.is_running(req.index) {
            drop(mgr);
            let payload = serde_json::to_string(&serde_json::json!({
                "index": req.index,
                "name": name,
                "status": "already_running"
            }))
            .unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        }

        // Start the source processor
        let stop_signal = Arc::new(AtomicBool::new(false));
        let handle = if let Some(ref start_fn) = start_source_fn {
            Some(start_fn(req.index, url.clone(), stop_signal.clone()))
        } else {
            None
        };

        mgr.running.insert(
            req.index,
            RunningSource {
                index: req.index,
                name: name.clone(),
                url: url.clone(),
                stop_signal,
                handle,
            },
        );

        // Update config file - reload to preserve any concurrent overlay changes
        drop(mgr);
        if let Ok(mut fresh_config) = load_rtsp_config(&config_path) {
            if !fresh_config.active_sources.contains(&req.index) {
                fresh_config.active_sources.push(req.index);
                fresh_config.active_sources.sort_unstable();
                let _ = save_rtsp_config(&config_path, &fresh_config);
            }
        }

        info!("Started source {} ({})", req.index, name);
        let payload = serde_json::to_string(&serde_json::json!({
            "index": req.index,
            "name": name,
            "status": "started"
        }))
        .unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // POST /api/sources/stop - stop processing a source (live, no restart needed)
    // Body: {"index": 3}
    if method == "POST" && path == "/api/sources/stop" {
        let req: SourceIndexRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => {
                return respond(
                    stream,
                    400,
                    "text/plain",
                    "Invalid JSON: expected {\"index\": N}",
                )
            }
        };

        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };

        if req.index < 1 || req.index > config.rtsp_links.len() {
            return respond(
                stream,
                400,
                "text/plain",
                &format!(
                    "Invalid index {}: must be between 1 and {}",
                    req.index,
                    config.rtsp_links.len()
                ),
            );
        }

        let name = source_display_name(&config.rtsp_links[req.index - 1]);

        let mut mgr = source_manager.lock().unwrap();
        let stopped = mgr.stop_source(req.index);
        drop(mgr);

        if stopped {
            // Update config file - reload to preserve any concurrent overlay changes
            if let Ok(mut fresh_config) = load_rtsp_config(&config_path) {
                fresh_config.active_sources.retain(|&i| i != req.index);
                let _ = save_rtsp_config(&config_path, &fresh_config);
            }

            info!("Stopped source {} ({})", req.index, name);
            let payload = serde_json::to_string(&serde_json::json!({
                "index": req.index,
                "name": name,
                "status": "stopped"
            }))
            .unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        } else {
            let payload = serde_json::to_string(&serde_json::json!({
                "index": req.index,
                "name": name,
                "status": "not_running"
            }))
            .unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        }
    }

    // POST /api/sources/active - set which sources are active (starts/stops as needed)
    // Body: {"indexes": [1, 3, 5]} - array of 1-based indexes
    if method == "POST" && path == "/api/sources/active" {
        let req: SetActiveSourcesRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => {
                return respond(
                    stream,
                    400,
                    "text/plain",
                    "Invalid JSON: expected {\"indexes\": [1, 2, ...]}",
                )
            }
        };

        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };

        // Validate all indexes
        for idx in &req.indexes {
            if *idx < 1 || *idx > config.rtsp_links.len() {
                return respond(
                    stream,
                    400,
                    "text/plain",
                    &format!(
                        "Invalid index {}: must be between 1 and {}",
                        idx,
                        config.rtsp_links.len()
                    ),
                );
            }
        }

        let mut desired: Vec<usize> = req.indexes.clone();
        desired.sort_unstable();
        desired.dedup();

        let mut mgr = source_manager.lock().unwrap();
        let currently_running = mgr.get_running_indexes();

        // Stop sources that should no longer be running
        let to_stop: Vec<usize> = currently_running
            .iter()
            .filter(|i| !desired.contains(i))
            .copied()
            .collect();
        for idx in &to_stop {
            mgr.stop_source(*idx);
        }

        // Start sources that should be running
        let to_start: Vec<usize> = desired
            .iter()
            .filter(|i| !currently_running.contains(i))
            .copied()
            .collect();

        for idx in &to_start {
            let url = config.rtsp_links[idx - 1].clone();
            let name = source_display_name(&url);
            let stop_signal = Arc::new(AtomicBool::new(false));
            let handle = if let Some(ref start_fn) = start_source_fn {
                Some(start_fn(*idx, url.clone(), stop_signal.clone()))
            } else {
                None
            };
            mgr.running.insert(
                *idx,
                RunningSource {
                    index: *idx,
                    name,
                    url,
                    stop_signal,
                    handle,
                },
            );
        }

        let final_running = mgr.get_running_indexes();
        drop(mgr);

        // Update config file - reload to preserve any concurrent overlay changes
        if let Ok(mut fresh_config) = load_rtsp_config(&config_path) {
            fresh_config.active_sources = final_running.clone();
            let _ = save_rtsp_config(&config_path, &fresh_config);
        }

        let names: Vec<String> = final_running
            .iter()
            .filter_map(|i| config.rtsp_links.get(i - 1).map(|u| source_display_name(u)))
            .collect();
        info!("Active sources updated: {:?} ({:?})", final_running, names);

        let payload = serde_json::to_string(&serde_json::json!({
            "active_sources": final_running,
            "names": names,
            "started": to_start,
            "stopped": to_stop
        }))
        .unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // GET /api/sources/active - get currently running sources
    if method == "GET" && path == "/api/sources/active" {
        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };
        let mgr = source_manager.lock().unwrap();
        let active = mgr.get_running_indexes();
        drop(mgr);

        let names: Vec<String> = active
            .iter()
            .filter_map(|i| config.rtsp_links.get(i - 1).map(|u| source_display_name(u)))
            .collect();
        let payload = serde_json::to_string(&serde_json::json!({
            "active_sources": active,
            "names": names
        }))
        .unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // GET /api/metrics - get current metrics for all active sources
    if method == "GET" && path == "/api/metrics" {
        // Return real metrics from the shared metrics map
        let metrics_guard = metrics_map.lock().unwrap();
        let payload = serde_json::to_string(&*metrics_guard).unwrap_or_else(|_| "{}".to_string());
        drop(metrics_guard);
        return respond(stream, 200, "application/json", &payload);
    }

    // POST /api/login
    if method == "POST" && path == "/api/login" {
        let req: LoginRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => {
                return respond(stream, 400, "text/plain", "Invalid JSON");
            }
        };

        let conn = match Connection::open(&db_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to open DB for login: {}", e);
                return respond(stream, 500, "text/plain", "Internal Server Error");
            }
        };

        let mut stmt = match conn.prepare("SELECT password_hash FROM users WHERE username = ?1") {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to prepare login statement: {}", e);
                return respond(stream, 500, "text/plain", "Internal Server Error");
            }
        };

        let mut rows = match stmt.query([&req.username]) {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to execute login query: {}", e);
                return respond(stream, 500, "text/plain", "Internal Server Error");
            }
        };

        if let Ok(Some(row)) = rows.next() {
            let password_hash: String = row.get(0).unwrap_or_default();
            // In a real app, verify hash. Here we just compare plain text as per instructions/current setup
            if password_hash == req.password {
                let payload = serde_json::json!({
                    "success": true,
                    "token": "dummy-token-123" 
                });
                return respond(stream, 200, "application/json", &payload.to_string());
            }
        }

        let payload = serde_json::json!({
            "success": false,
            "error": "Invalid credentials"
        });
        return respond(stream, 401, "application/json", &payload.to_string());
    }

    // GET /api/jobs - list all upload jobs and their status
    if method == "GET" && path == "/api/jobs" {
        let job_map = jobs_map.lock().unwrap();
        let jobs: Vec<&UploadJob> = job_map.values().collect();
        let payload = serde_json::to_string(&serde_json::json!({ "jobs": jobs }))
            .unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // GET /api/jobs/{job_id} - get status of a specific upload job
    if method == "GET" && path.starts_with("/api/jobs/") {
        let job_id = &path["/api/jobs/".len()..];
        let job_map = jobs_map.lock().unwrap();
        if let Some(job) = job_map.get(job_id) {
            let payload = serde_json::to_string(job).unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        }
        return respond(stream, 404, "text/plain", "Job not found");
    }

    // GET /api/outputs - list available output files in runs/drone_analysis/
    if method == "GET" && path == "/api/outputs" {
        let out_dir = db_path.parent().unwrap_or(std::path::Path::new("."));
        let mut files: Vec<serde_json::Value> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(out_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".mp4") || name.ends_with(".mkv") {
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    let modified = entry
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    files.push(serde_json::json!({
                        "name": name,
                        "size": size,
                        "modified": modified
                    }));
                }
            }
        }
        // Sort by modified desc (most recent first)
        files.sort_by(|a, b| {
            let ma = a["modified"].as_u64().unwrap_or(0);
            let mb = b["modified"].as_u64().unwrap_or(0);
            mb.cmp(&ma)
        });
        let payload = serde_json::to_string(&serde_json::json!({ "files": files }))
            .unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // GET /api/download?file={filename} - download an output file
    if method == "GET" && path.starts_with("/api/download") {
        let query_part = path.split('?').nth(1).unwrap_or("");
        let mut filename = String::new();
        for pair in query_part.split('&') {
            if let Some(rest) = pair.strip_prefix("file=") {
                filename = url_decode(rest);
            }
        }

        if filename.is_empty() {
            return respond(stream, 400, "text/plain", "Missing file parameter");
        }

        // Path traversal protection
        if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
            return respond(stream, 400, "text/plain", "Invalid filename");
        }

        let out_dir = db_path.parent().unwrap_or(std::path::Path::new("."));
        let file_path = out_dir.join(&filename);

        if !file_path.exists() {
            return respond(stream, 404, "text/plain", "File not found");
        }

        let content_type = if filename.ends_with(".mkv") {
            "video/x-matroska"
        } else {
            "video/mp4"
        };

        let file_size = std::fs::metadata(&file_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let mut file = match File::open(&file_path) {
            Ok(f) => f,
            Err(e) => {
                warn!("Failed to open download file: {}", e);
                return respond(stream, 500, "text/plain", "Failed to open file");
            }
        };

        // Write HTTP headers
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nContent-Disposition: attachment; filename=\"{}\"\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
            content_type, file_size, filename
        );
        stream.write_all(header.as_bytes())?;

        // Stream file in 64KB chunks
        let mut buf = vec![0u8; 65536];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            stream.write_all(&buf[..n])?;
        }
        return Ok(());
    }

    debug!("Control API: 404 for {} {}", method, path);
    respond(stream, 404, "text/plain", "Not Found")
}

/// Save overlay state to the RTSP config file for persistence
fn persist_overlay_to_config(
    config_path: &PathBuf,
    stream_name: &str,
    state: OverlayState,
) -> anyhow::Result<()> {
    let mut config = load_rtsp_config(config_path)?;
    config
        .overlays
        .insert(stream_name.to_string(), state.into());
    save_rtsp_config(config_path, &config)?;
    debug!(
        "Persisted overlay config for '{}' to {}",
        stream_name,
        config_path.display()
    );
    Ok(())
}

fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let status_line = match status {
        200 => "HTTP/1.1 200 OK",
        400 => "HTTP/1.1 400 Bad Request",
        401 => "HTTP/1.1 401 Unauthorized",
        404 => "HTTP/1.1 404 Not Found",
        _ => "HTTP/1.1 500 Internal Server Error",
    };
    let response = format!(
        "{status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{body}",
        body.as_bytes().len()
    );
    stream.write_all(response.as_bytes())?;
    Ok(())
}

fn url_decode(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            if let (Some(h1), Some(h2)) = (chars.next(), chars.next()) {
                if let Ok(byte) = u8::from_str_radix(&format!("{}{}", h1, h2), 16) {
                    out.push(byte as char);
                }
            }
        } else if c == '+' {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}
