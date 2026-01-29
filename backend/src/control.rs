use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::helpers::{load_rtsp_config, mask_rtsp_credentials, save_rtsp_config, source_display_name, OverlayConfig};

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

pub type OverlayMap = Arc<Mutex<HashMap<String, OverlayState>>>;

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
pub type StartSourceFn = Arc<dyn Fn(usize, String, Arc<AtomicBool>) -> JoinHandle<()> + Send + Sync>;

/// Shared state for the control server including config path for persistence
pub struct ControlServerState {
    pub overlays: OverlayMap,
    pub config_path: PathBuf,
    pub source_manager: SharedSourceManager,
    pub start_source_fn: Option<StartSourceFn>,
}

pub type SharedControlState = Arc<Mutex<ControlServerState>>;

pub fn start_control_server(
    addr: &str,
    overlays: OverlayMap,
    config_path: PathBuf,
    source_manager: SharedSourceManager,
    start_source_fn: StartSourceFn,
) -> thread::JoinHandle<()> {
    let addr = addr.to_string();
    let state = Arc::new(Mutex::new(ControlServerState {
        overlays,
        config_path,
        source_manager,
        start_source_fn: Some(start_source_fn),
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
    let mut buf = [0u8; 4096];
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        if data.windows(4).any(|w| w == b"\r\n\r\n") || data.len() > 65536 {
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
    for line in lines {
        if let Some(rest) = line.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse::<usize>().unwrap_or(0);
        }
    }

    let mut body = data[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
    }

    debug!("Control API: {} {}", method, path);

    let ctrl_state = state.lock().unwrap();
    let overlays = ctrl_state.overlays.clone();
    let config_path = ctrl_state.config_path.clone();
    let source_manager = ctrl_state.source_manager.clone();
    let start_source_fn = ctrl_state.start_source_fn.clone();
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
            warn!("Control API: Stream '{}' not found. Available: {:?}",
                  stream_name, map.keys().collect::<Vec<_>>());
            return respond(stream, 404, "text/plain", "Not Found");
        }

        if method == "POST" {
            let update: OverlayUpdate = serde_json::from_slice(&body).unwrap_or(OverlayUpdate {
                heatmap: None,
                trails: None,
                bboxes: None,
            });
            debug!("POST overlay update for '{}': heatmap={:?}, trails={:?}, bboxes={:?}",
                   stream_name, update.heatmap, update.trails, update.bboxes);

            let next = {
                let mut map = overlays.lock().unwrap();

                // Check if stream exists
                if !map.contains_key(stream_name) {
                    warn!("Control API: Stream '{}' not found for update. Available: {:?}",
                          stream_name, map.keys().collect::<Vec<_>>());
                }

                let current = map.get(stream_name).copied().unwrap_or_default();
                let next = OverlayState {
                    heatmap: update.heatmap.unwrap_or(current.heatmap),
                    trails: update.trails.unwrap_or(current.trails),
                    bboxes: update.bboxes.unwrap_or(current.bboxes),
                };
                map.insert(stream_name.to_string(), next);
                info!("[{}] Overlay state changed: heatmap={}, trails={}, bboxes={}",
                      stream_name, next.heatmap, next.trails, next.bboxes);
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

    // POST /api/sources/start - start processing a source (live, no restart needed)
    // Body: {"index": 3}
    if method == "POST" && path == "/api/sources/start" {
        let req: SourceIndexRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => return respond(stream, 400, "text/plain", "Invalid JSON: expected {\"index\": N}"),
        };

        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };

        if req.index < 1 || req.index > config.rtsp_links.len() {
            return respond(stream, 400, "text/plain", &format!(
                "Invalid index {}: must be between 1 and {}", req.index, config.rtsp_links.len()
            ));
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
            })).unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        }

        // Start the source processor
        let stop_signal = Arc::new(AtomicBool::new(false));
        let handle = if let Some(ref start_fn) = start_source_fn {
            Some(start_fn(req.index, url.clone(), stop_signal.clone()))
        } else {
            None
        };

        mgr.running.insert(req.index, RunningSource {
            index: req.index,
            name: name.clone(),
            url: url.clone(),
            stop_signal,
            handle,
        });

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
        })).unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    // POST /api/sources/stop - stop processing a source (live, no restart needed)
    // Body: {"index": 3}
    if method == "POST" && path == "/api/sources/stop" {
        let req: SourceIndexRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => return respond(stream, 400, "text/plain", "Invalid JSON: expected {\"index\": N}"),
        };

        let config = match load_rtsp_config(&config_path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to load config: {}", e);
                return respond(stream, 500, "text/plain", "Failed to load config");
            }
        };

        if req.index < 1 || req.index > config.rtsp_links.len() {
            return respond(stream, 400, "text/plain", &format!(
                "Invalid index {}: must be between 1 and {}", req.index, config.rtsp_links.len()
            ));
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
            })).unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        } else {
            let payload = serde_json::to_string(&serde_json::json!({
                "index": req.index,
                "name": name,
                "status": "not_running"
            })).unwrap_or_else(|_| "{}".to_string());
            return respond(stream, 200, "application/json", &payload);
        }
    }

    // POST /api/sources/active - set which sources are active (starts/stops as needed)
    // Body: {"indexes": [1, 3, 5]} - array of 1-based indexes
    if method == "POST" && path == "/api/sources/active" {
        let req: SetActiveSourcesRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(_) => return respond(stream, 400, "text/plain", "Invalid JSON: expected {\"indexes\": [1, 2, ...]}"),
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
                return respond(stream, 400, "text/plain", &format!(
                    "Invalid index {}: must be between 1 and {}", idx, config.rtsp_links.len()
                ));
            }
        }

        let mut desired: Vec<usize> = req.indexes.clone();
        desired.sort_unstable();
        desired.dedup();

        let mut mgr = source_manager.lock().unwrap();
        let currently_running = mgr.get_running_indexes();

        // Stop sources that should no longer be running
        let to_stop: Vec<usize> = currently_running.iter()
            .filter(|i| !desired.contains(i))
            .copied()
            .collect();
        for idx in &to_stop {
            mgr.stop_source(*idx);
        }

        // Start sources that should be running
        let to_start: Vec<usize> = desired.iter()
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
            mgr.running.insert(*idx, RunningSource {
                index: *idx,
                name,
                url,
                stop_signal,
                handle,
            });
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
        })).unwrap_or_else(|_| "{}".to_string());
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
        })).unwrap_or_else(|_| "{}".to_string());
        return respond(stream, 200, "application/json", &payload);
    }

    debug!("Control API: 404 for {} {}", method, path);
    respond(stream, 404, "text/plain", "Not Found")
}

/// Save overlay state to the RTSP config file for persistence
fn persist_overlay_to_config(config_path: &PathBuf, stream_name: &str, state: OverlayState) -> anyhow::Result<()> {
    let mut config = load_rtsp_config(config_path)?;
    config.overlays.insert(stream_name.to_string(), state.into());
    save_rtsp_config(config_path, &config)?;
    debug!("Persisted overlay config for '{}' to {}", stream_name, config_path.display());
    Ok(())
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) -> std::io::Result<()> {
    let status_line = match status {
        200 => "HTTP/1.1 200 OK",
        400 => "HTTP/1.1 400 Bad Request",
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
