// src/main.rs

use anyhow::{anyhow, Result};
use chrono::Local;
use jamtrack_rs::{Object, Rect as JamRect};
use log::{debug, error, info, warn};
use opencv::highgui;
use opencv::prelude::VideoWriterTrait;
use std::{
    env,
    fs,
    path::PathBuf,
};
use std::time::Instant;
use std::thread;
use ultralytics_inference::Device;
#[cfg(not(feature = "trt_engine"))]
use ultralytics_inference::{InferenceConfig, YOLOModel};

mod analytics;
use analytics::*;
mod helpers;
use helpers::*;
mod state;
use state::*;
mod mediamtx;
mod control;
use control::{MetricsMap, OverlayMap, OverlayState, SourceManager, SourceMetrics, SharedSourceManager, RunningSource};
mod db;
use db::Db;
#[cfg(feature = "trt_engine")]
mod trt;
#[cfg(feature = "trt_engine")]
mod trt_yolo;

const EMA_ALPHA: f32 = 0.2;
const DB_AGG_INTERVAL_SECS: u64 = 60;

fn analyze_frame(
    state: &mut SourceState,
    frame_idx: usize,
    objects: &[Object],
    det_classes: &[i64],
    det_confs: &[f32],
) -> Result<(Vec<Detection>, i32, i32, i32, i32, i32, i32, i32)> {
    let tracked = apply_tracks(
        objects.to_vec(),
        det_classes,
        det_confs,
        &mut state.tracks,
        &mut state.next_local_id,
        frame_idx,
        state.fps,
    );

    decay_heatmap(&mut state.heatmap)?;
    update_heatmap(&mut state.heatmap, &state.tracks)?;

    let mut counts = [0u32; 4];
    let mut area_sum = 0.0f64;
    for det in &tracked {
        counts[det.speed.bucket()] += 1;
        let w = (det.bbox.x2 - det.bbox.x1).max(0.0) as f64;
        let h = (det.bbox.y2 - det.bbox.y1).max(0.0) as f64;
        area_sum += w * h;
    }
    let total = tracked.len() as f64;
    let pct = |v: u32| if total > 0.0 { ((v as f64 / total) * 100.0).round() as i32 } else { 0 };
    let stalled_pct = pct(counts[0]);
    let slow_pct = pct(counts[1]);
    let medium_pct = pct(counts[2]);
    let fast_pct = pct(counts[3]);

    let frame_area = (state.w as f64).max(1.0) * (state.h as f64).max(1.0);
    let density = ((area_sum / frame_area) * 100.0).clamp(0.0, 100.0) as f32;
    if state.traffic_density_ema <= 0.0 {
        state.traffic_density_ema = density;
    } else {
        state.traffic_density_ema = state.traffic_density_ema * (1.0 - EMA_ALPHA) + density * EMA_ALPHA;
    }
    let traffic_density_pct = state.traffic_density_ema.round().clamp(0.0, 100.0) as i32;

    let mobility = if total > 0.0 {
        let w_stalled = 10.0;
        let w_slow = 40.0;
        let w_medium = 70.0;
        let w_fast = 95.0;
        let score = counts[0] as f64 * w_stalled
            + counts[1] as f64 * w_slow
            + counts[2] as f64 * w_medium
            + counts[3] as f64 * w_fast;
        (score / total).clamp(0.0, 100.0) as f32
    } else {
        0.0
    };
    if state.mobility_index_ema <= 0.0 {
        state.mobility_index_ema = mobility;
    } else {
        state.mobility_index_ema = state.mobility_index_ema * (1.0 - EMA_ALPHA) + mobility * EMA_ALPHA;
    }
    let mobility_index_pct = state.mobility_index_ema.round().clamp(0.0, 100.0) as i32;

    let congestion = ((traffic_density_pct as f32 * 0.6)
        + ((100 - mobility_index_pct) as f32 * 0.4))
        .round()
        .clamp(0.0, 100.0) as i32;

    Ok((
        tracked,
        congestion,
        traffic_density_pct,
        mobility_index_pct,
        stalled_pct,
        slow_pct,
        medium_pct,
        fast_pct,
    ))
}

// entry point
fn main() -> Result<()> {
    // Initialize logger - use RUST_LOG env var to control level (e.g. RUST_LOG=debug)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    info!("=== IRIS Drone Analytics Starting ===");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    process_video()
}

fn process_source(
    source: String,
    opts: CliOptions,
    out_dir: &PathBuf,
    data_dir: &PathBuf,
    base_device: Device,
    overlays: OverlayMap,
    metrics_map: MetricsMap,
    show_ui: bool,
    stop_signal: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    skip_mediamtx: bool,
) -> Result<String> {
    let source_label = source_display_name(&source);
    info!("[{}] === Starting source processing ===", source_label);

    info!("[{}] Opening analytics database...", source_label);
    let mut db = Db::open(&out_dir.join("analytics.db"))?;
    info!("[{}] Database opened: {}", source_label, out_dir.join("analytics.db").display());

    let mut inference_device = base_device;
    let engine_path = data_dir.join("yolov11n-visdrone.engine");
    if engine_path.exists() && !cfg!(feature = "trt_engine") {
        return Err(anyhow!(
            "TensorRT engine found at {} but binary was built without `trt_engine` feature",
            engine_path.display()
        ));
    }


    #[cfg(feature = "trt_engine")]
    let (model_path, mut trt_model) = {
        let engine_path = data_dir.join("yolov11n-visdrone.engine");
        let onnx_path = data_dir.join("yolov11n-visdrone.onnx");

        info!("[{}] Loading TensorRT model...", source_label);
        info!("[{}] Engine path: {}", source_label, engine_path.display());
        info!("[{}] ONNX path: {}", source_label, onnx_path.display());

        let force_rebuild = std::env::var("FORCE_REBUILD").is_ok();

        if (!engine_path.exists() || force_rebuild) && onnx_path.exists() {
            info!("[{}] Building TensorRT engine from ONNX (this may take a few minutes)...", source_label);
            trt::TrtRunner::build_engine(
                onnx_path.to_str().unwrap(),
                engine_path.to_str().unwrap()
            )?;
            info!("[{}] TensorRT engine built successfully", source_label);
        }

        if !engine_path.exists() {
            error!("[{}] TensorRT engine not found: {}", source_label, engine_path.display());
            return Err(anyhow!(
                "TensorRT engine not found and could not be built: {}",
                engine_path.display()
            ));
        }
        info!("[{}] Loading TensorRT engine...", source_label);
        let model = trt_yolo::TrtYolo::new(engine_path.to_str().unwrap())?;
        info!("[{}] TensorRT model loaded successfully", source_label);
        (engine_path, model)
    };

    #[cfg(not(feature = "trt_engine"))]
    let (model_path, mut model) = {
        let default_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = opts.threads.unwrap_or(default_threads);
        info!("[{}] Inference threads: {}", source_label, threads);

        let build_cfg = |device: Device| {
            InferenceConfig::new()
                .with_confidence(CONF_THRESH)
                .with_iou(0.45)
                .with_max_det(500)
                .with_threads(threads)
                .with_device(device)
        };

        let onnx_path = data_dir.join("yolov11n-visdrone.onnx");
        let model_path = onnx_path.clone();
        warn!(
            "[{}] TensorRT not available; falling back to ONNX model at {}",
            source_label,
            model_path.display()
        );


        if !model_path.exists() {
            error!("[{}] Model file not found: {}", source_label, model_path.display());
            return Err(anyhow!("Model not found: {}", model_path.display()));
        }
        info!("[{}] Model file exists, loading...", source_label);

        info!("[{}] Attempting to load model with device: {}", source_label, format_device_label(&inference_device));
        let model = match YOLOModel::load_with_config(
            &model_path,
            build_cfg(inference_device.clone()),
        ) {
            Ok(model) => {
                info!("[{}] Model loaded successfully on {}", source_label, format_device_label(&inference_device));
                model
            }
            Err(err) => {
                if matches!(inference_device, Device::TensorRt(_)) {
                    warn!("[{}] TensorRT load failed ({}); falling back to CUDA device 0", source_label, err);
                    inference_device = Device::Cuda(0);
                    match YOLOModel::load_with_config(
                        &model_path,
                        build_cfg(inference_device.clone()),
                    ) {
                        Ok(model) => {
                            info!("[{}] Model loaded successfully on CUDA", source_label);
                            model
                        }
                        Err(err) => {
                            warn!("[{}] CUDA load failed ({}); falling back to CPU", source_label, err);
                            inference_device = Device::Cpu;
                            let model = YOLOModel::load_with_config(
                                &model_path,
                                build_cfg(inference_device.clone()),
                            )?;
                            info!("[{}] Model loaded successfully on CPU", source_label);
                            model
                        }
                    }
                } else {
                    error!("[{}] Failed to load model: {}", source_label, err);
                    return Err(err.into());
                }
            }
        };
        (model_path, model)
    };

    let uses_engine = cfg!(feature = "trt_engine") && model_path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("engine"))
        .unwrap_or(false);

    let model_kind = if uses_engine {
        "TensorRT engine"
    } else {
        "ONNX model"
    };
    info!("[{}] Model in use: {} ({})", source_label, model_kind, model_path.display());

    if show_ui {
        info!("[{}] Creating display window...", source_label);
        highgui::named_window("Drone Analytics", highgui::WINDOW_NORMAL)?;
    } else {
        info!("[{}] Running in headless mode (no display)", source_label);
    }

    info!("[{}] Connecting to video source: {}", source_label, mask_rtsp_credentials(&source));
    let (mut state, cap) = init_source_state(
        &source,
        out_dir,
        opts.output,
        inference_device.clone(),
        opts.stride,
        skip_mediamtx,
    )?;
    info!("[{}] Video source connected successfully", source_label);
    info!("[{}] Video properties: {}x{} @ {:.1} FPS", source_label, state.w, state.h, state.fps);
    info!("[{}] Processing stride: {} (effective FPS: {:.1})", source_label, state.stride, state.out_fps);

    let is_rtsp = is_rtsp_source(&source);
    if is_rtsp {
        info!("[{}] Source type: RTSP stream", source_label);
    } else {
        info!("[{}] Source type: Local file", source_label);
    }

    info!("[{}] Starting capture thread (buffer size: {} frames)...", source_label, opts.io_buffer);
    let mut capture = Some(start_capture_thread(cap, state.stride, opts.io_buffer, is_rtsp));
    info!("[{}] Capture thread started", source_label);

    if show_ui {
        if is_rtsp {
            highgui::resize_window("Drone Analytics", state.w, state.h)?;
        } else {
            highgui::resize_window("Drone Analytics", 1280, 720)?;
        }
    }

    if state.mediamtx.is_some() {
        info!("[{}] RTSP republishing enabled", source_label);
    }

    info!("[{}] Creating analytics session...", source_label);
    let session_started = Local::now().to_rfc3339();
    let overlay_key = source_label.clone();
    info!("[{}] Registering overlay control key: '{}'", source_label, overlay_key);
    {
        let mut map = overlays.lock().unwrap();
        map.entry(overlay_key.clone()).or_insert(OverlayState {
            heatmap: opts.show_heatmap,
            trails: opts.show_trails,
            bboxes: opts.show_bboxes,
        });
        debug!("[{}] Available overlay keys: {:?}", source_label, map.keys().collect::<Vec<_>>());
    }
    let model_label = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let config_json = format!(
        "{{\"heatmap\":{},\"trails\":{},\"bboxes\":{},\"threads\":{},\"device\":\"{}\"}}",
        opts.show_heatmap,
        opts.show_trails,
        opts.show_bboxes,
        opts.threads.unwrap_or(0),
        format_device_label(&inference_device)
    );
    let session_id = db.create_session(
        &session_started,
        &source,
        &source_label,
        state.fps,
        state.w,
        state.h,
        model_label,
        &format_device_label(&inference_device),
        &config_json,
    )?;
    info!("[{}] Session created (ID: {})", source_label, session_id);
    let mut last_flush = Instant::now();
    let mut agg = MinuteAgg::default();
    let mut show_heatmap = opts.show_heatmap;
    let mut show_trails = opts.show_trails;
    let mut show_bboxes = opts.show_bboxes;

    info!("[{}] === Starting inference loop ===", source_label);
    info!("[{}] Overlays - heatmap: {}, trails: {}, bboxes: {}", source_label, show_heatmap, show_trails, show_bboxes);
    let mut frame_count = 0usize;
    let mut last_log_time = Instant::now();
    let log_interval_secs = 10;
    let mut last_metrics_update = Instant::now();
    let mut metrics_frame_count = 0usize;

    loop {
        // Check for external stop signal
        if let Some(ref stop) = stop_signal {
            if stop.load(std::sync::atomic::Ordering::SeqCst) {
                info!("[{}] Stop signal received, shutting down...", source_label);
                break;
            }
        }

        let Some(rx) = capture.as_ref().map(|worker| worker.rx.clone()) else {
            break;
        };
        let mut action: Option<i32> = None;
        while let Ok(packet) = rx.recv() {
            // Check stop signal on each frame
            if let Some(ref stop) = stop_signal {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    info!("[{}] Stop signal received, shutting down...", source_label);
                    action = Some(113); // 'q' to quit
                    break;
                }
            }
            let mut frame = packet.frame;
            let frame_idx = packet.idx;

            let mut objects: Vec<Object> = Vec::new();
            let mut classes: Vec<i64> = Vec::new();
            let mut det_confs: Vec<f32> = Vec::new();
            #[cfg(feature = "trt_engine")]
            {
                let dets = trt_model.predict(&frame)?;
                for det in dets {
                    let cid = det.class_id;
                    if !TARGET_CLASSES.contains(&cid) {
                        continue;
                    }
                    objects.push(Object::new(
                        JamRect::from_xyxy(det.x1, det.y1, det.x2, det.y2),
                        det.score,
                        None,
                    ));
                    classes.push(cid);
                    det_confs.push(det.score);
                }
            }

            #[cfg(not(feature = "trt_engine"))]
            let dets = {
                let arr = mat_to_array3_rgb(&frame)?;
                model.predict_array(&arr, source_label.clone())?
            };
            #[cfg(not(feature = "trt_engine"))]
            for result in dets.iter() {
                let Some(boxes) = &result.boxes else {
                    continue;
                };
                let data = &boxes.data;
                for i in 0..data.nrows() {
                    let x1 = data[[i, 0]];
                    let y1 = data[[i, 1]];
                    let x2 = data[[i, 2]];
                    let y2 = data[[i, 3]];
                    let conf = data[[i, 4]];
                    let cid = data[[i, 5]] as i64;
                    if !TARGET_CLASSES.contains(&cid) {
                        continue;
                    }
                    objects.push(Object::new(
                        JamRect::from_xyxy(x1, y1, x2, y2),
                        conf,
                        None,
                    ));
                    classes.push(cid);
                    det_confs.push(conf);
                }
            }

            let (dets, congestion, traffic_density_pct, mobility_index_pct, stalled_pct, slow_pct, medium_pct, fast_pct) =
                analyze_frame(
                    &mut state,
                    frame_idx,
                    &objects,
                    &classes,
                    &det_confs,
                )?;
            agg.push(
                frame_idx,
                chrono::Local::now().timestamp_millis(),
                dets.len(),
                congestion,
                traffic_density_pct,
                mobility_index_pct,
                stalled_pct,
                slow_pct,
                medium_pct,
                fast_pct,
            );

            // Update shared metrics frequently for real-time API access
            metrics_frame_count += 1;
            if last_metrics_update.elapsed().as_millis() >= 500 {
                let elapsed = last_metrics_update.elapsed().as_secs_f32();
                let fps_actual = metrics_frame_count as f32 / elapsed;
                {
                    let mut metrics = metrics_map.lock().unwrap();
                    metrics.insert(source_label.clone(), SourceMetrics {
                        fps: fps_actual,
                        congestion_index: congestion,
                        traffic_density: traffic_density_pct,
                        mobility_index: mobility_index_pct,
                        stalled_pct,
                        slow_pct,
                        medium_pct,
                        fast_pct,
                        detection_count: dets.len(),
                    });
                }
                metrics_frame_count = 0;
                last_metrics_update = Instant::now();
            }

            // Periodic status logging
            frame_count += 1;
            if last_log_time.elapsed().as_secs() >= log_interval_secs {
                let elapsed = last_log_time.elapsed().as_secs_f32();
                let fps_actual = frame_count as f32 / elapsed;
                info!("[{}] Frame {} | {} detections | congestion: {}% | FPS: {:.1}",
                      source_label, frame_idx, dets.len(), congestion, fps_actual);
                debug!("[{}] Traffic density: {}% | Mobility: {}% | Speed dist: stalled={}% slow={}% medium={}% fast={}%",
                       source_label, traffic_density_pct, mobility_index_pct, stalled_pct, slow_pct, medium_pct, fast_pct);
                frame_count = 0;
                last_log_time = Instant::now();
            }

            if last_flush.elapsed().as_secs() >= DB_AGG_INTERVAL_SECS {
                if let Some(row) = agg.take_averages() {
                    debug!("[{}] Flushing aggregated metrics to database", source_label);
                    db.insert_frame_metrics(
                        session_id,
                        row.frame_idx,
                        row.ts_ms,
                        row.detections,
                        row.congestion,
                        row.traffic_density,
                        row.mobility_index,
                        row.stalled_pct,
                        row.slow_pct,
                        row.medium_pct,
                        row.fast_pct,
                        None,
                    )?;
                }
                last_flush = Instant::now();
            }

            if let Some(overlay_state) = overlays.lock().unwrap().get(&overlay_key).copied() {
                // Log when overlay state changes
                if overlay_state.heatmap != show_heatmap ||
                   overlay_state.trails != show_trails ||
                   overlay_state.bboxes != show_bboxes {
                    info!("[{}] Overlay state updated: heatmap={}, trails={}, bboxes={}",
                          source_label, overlay_state.heatmap, overlay_state.trails, overlay_state.bboxes);
                }
                show_heatmap = overlay_state.heatmap;
                show_trails = overlay_state.trails;
                show_bboxes = overlay_state.bboxes;
            }

            let out = &mut frame;
            if show_heatmap {
                overlay_heatmap(&state.heatmap, out)?;
            }
            if show_trails {
                draw_trails(out, &state.tracks)?;
            }
            if show_bboxes {
                draw_detections(out, &dets)?;
            }
            if let Some(pub_) = state.mediamtx.as_mut() {
                pub_.send(out)?;
            }
            if state.writer_active {
                if let Some(limit) = state.write_limit {
                    if state.frames_written >= limit {
                        state.writer.release()?;
                        state.writer_active = false;
                    }
                }
                if state.writer_active {
                    state.writer.write(out)?;
                    state.frames_written += 1;
                }
            }
            if show_ui {
                highgui::imshow("Drone Analytics", out)?;
            }
            let key = if show_ui { highgui::wait_key(1)? } else { -1 };
            if key == 113 {
                action = Some(key);
                break;
            }
            if key == 104 {
                show_heatmap = !show_heatmap;
                overlays.lock().unwrap().insert(overlay_key.clone(), OverlayState {
                    heatmap: show_heatmap,
                    trails: show_trails,
                    bboxes: show_bboxes,
                });
            }
            if key == 116 {
                show_trails = !show_trails;
                overlays.lock().unwrap().insert(overlay_key.clone(), OverlayState {
                    heatmap: show_heatmap,
                    trails: show_trails,
                    bboxes: show_bboxes,
                });
            }
            if key == 98 {
                show_bboxes = !show_bboxes;
                overlays.lock().unwrap().insert(overlay_key.clone(), OverlayState {
                    heatmap: show_heatmap,
                    trails: show_trails,
                    bboxes: show_bboxes,
                });
            }
        }
        if let Some(key) = action {
            if let Some(worker) = capture.take() {
                info!("[{}] Stopping capture thread...", source_label);
                worker.stop()?;
            }
            if key == 113 {
                info!("[{}] User requested quit (q key)", source_label);
                break;
            }
        } else {
            if let Some(worker) = capture.take() {
                info!("[{}] Video stream ended, stopping capture thread...", source_label);
                worker.stop()?;
            }
            break;
        }
    }

    info!("[{}] === Inference loop ended ===", source_label);
    let session_ended = Local::now().to_rfc3339();
    info!("[{}] Closing session (ID: {})...", source_label, session_id);
    let _ = db.finish_session(session_id, &session_ended);
    info!("[{}] Session closed at {}", source_label, session_ended);

    if show_ui {
        highgui::destroy_all_windows()?;
    }

    info!("[{}] === Source processing complete ===", source_label);
    Ok(state.out_path.to_string_lossy().to_string())
}


// full pipeline
fn process_video() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data");
    info!("Data directory: {}", data_dir.display());

    let default_input = data_dir.join("drone8.mp4");
    let rtsp_config = data_dir.join("rtsp_links.yml");

    info!("Loading RTSP configuration from: {}", rtsp_config.display());
    let config = load_rtsp_config(&rtsp_config)?;
    let rtsp_links = config.rtsp_links.clone();
    let saved_overlays = config.overlays.clone();

    if rtsp_links.is_empty() {
        info!("No RTSP links configured");
    } else {
        info!("Found {} RTSP link(s):", rtsp_links.len());
        for (i, link) in rtsp_links.iter().enumerate() {
            let masked = mask_rtsp_credentials(link);
            info!("  [{}] {}", i + 1, masked);
        }
    }
    if !saved_overlays.is_empty() {
        info!("Found {} saved overlay config(s):", saved_overlays.len());
        for (name, cfg) in &saved_overlays {
            info!("  {} -> heatmap={}, trails={}, bboxes={}", name, cfg.heatmap, cfg.trails, cfg.bboxes);
        }
    }

    info!("Parsing command-line arguments...");
    let mut opts = parse_args()?;
    debug!("CLI options: use_rtsp={}, indexes={:?}, headless={}, stride={}",
           opts.use_rtsp, opts.rtsp_indexes, opts.headless, opts.stride);
    if opts.device.is_none() && cfg!(target_os = "macos") {
        opts.device = Some(Device::Mps);
        info!("Auto-selected device: MPS (macOS)");
    } else if opts.device.is_none() {
        if cfg!(feature = "tensorrt") {
            opts.device = Some(Device::TensorRt(0));
            info!("Auto-selected device: TensorRT (GPU 0)");
        } else {
            opts.device = Some(Device::Cuda(0));
            info!("Auto-selected device: CUDA (GPU 0)");
        }
    } else {
        info!("Using device: {}", format_device_label(opts.device.as_ref().unwrap()));
    }

    let has_source = opts.source.is_some();
    let sources = match opts.source.as_ref() {
        Some(arg) => vec![arg.clone()],
        None => {
            if opts.use_rtsp {
                if rtsp_links.is_empty() {
                    return Err(anyhow!("No RTSP links found in {}", rtsp_config.display()));
                }
                rtsp_links
            } else if !rtsp_links.is_empty() {
                rtsp_links
            } else {
                vec![default_input.to_string_lossy().to_string()]
            }
        }
    };

    let mut sources = sources;
    sources.retain(|s| !s.trim().is_empty());
    if sources.is_empty() {
        info!("No sources specified, using default: {}", default_input.display());
        sources.push(default_input.to_string_lossy().to_string());
    }

    let out_dir = root.join("runs/drone_analysis");
    info!("Output directory: {}", out_dir.display());
    fs::create_dir_all(&out_dir)?;
    info!("Output directory created/verified");

    // Determine which sources to start initially
    let selected_sources: Vec<String> = if has_source {
        // CLI provided a specific source - use only that
        vec![opts.source.clone().unwrap()]
    } else if !opts.rtsp_indexes.is_empty() {
        // CLI provided specific indexes
        let mut indexes = opts.rtsp_indexes.clone();
        indexes.sort_unstable();
        indexes.dedup();
        indexes
            .into_iter()
            .filter_map(|idx| sources.get(idx).cloned())
            .collect()
    } else if !config.active_sources.is_empty() {
        // Use active_sources from config (1-based indexes)
        info!("Using {} active source(s) from config: {:?}",
              config.active_sources.len(), config.active_sources);
        config.active_sources
            .iter()
            .filter_map(|i| sources.get(i.saturating_sub(1)).cloned())
            .collect()
    } else {
        // No sources configured - start empty, wait for API calls
        info!("No active sources configured. Use API to start drones.");
        Vec::new()
    };

    let inference_device = opts.device.clone().unwrap_or(Device::Cpu);
    let has_display = env::var_os("DISPLAY").is_some() || env::var_os("WAYLAND_DISPLAY").is_some();
    let show_ui = has_display && !opts.headless && selected_sources.len() == 1;

    info!("Display available: {}", has_display);
    info!("UI mode: {}", if show_ui { "enabled" } else { "headless" });

    if !show_ui && has_display && !opts.headless && selected_sources.len() > 1 {
        warn!("Multiple sources selected; running headless and disabling UI controls.");
    }

    info!("Selected {} source(s) for processing:", selected_sources.len());
    for (i, src) in selected_sources.iter().enumerate() {
        let label = source_display_name(src);
        let masked = mask_rtsp_credentials(src);
        info!("  [{}] {} ({})", i + 1, label, masked);
    }

    let overlays: OverlayMap = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let metrics_map: MetricsMap = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    // Initialize overlays for all available sources
    for url in &sources {
        let key = source_display_name(url);
        let saved = config.overlays.get(&key);
        overlays.lock().unwrap().insert(
            key,
            OverlayState {
                heatmap: saved.map(|o| o.heatmap).unwrap_or(opts.show_heatmap),
                trails: saved.map(|o| o.trails).unwrap_or(opts.show_trails),
                bboxes: saved.map(|o| o.bboxes).unwrap_or(opts.show_bboxes),
            },
        );
    }

    // Create the source manager
    let source_manager: SharedSourceManager = std::sync::Arc::new(
        std::sync::Mutex::new(SourceManager::new(sources.clone()))
    );

    // Create the callback for starting sources dynamically
    let opts_for_callback = opts.clone();
    let out_dir_for_callback = out_dir.clone();
    let data_dir_for_callback = data_dir.clone();
    let device_for_callback = inference_device.clone();
    let overlays_for_callback = overlays.clone();
    let metrics_for_callback = metrics_map.clone();
    let _sources_for_callback = sources.clone(); // Available for future use

    let start_source_fn: control::StartSourceFn = std::sync::Arc::new(move |index: usize, url: String, stop_signal: std::sync::Arc<std::sync::atomic::AtomicBool>| {
        let opts = opts_for_callback.clone();
        let out_dir = out_dir_for_callback.clone();
        let data_dir = data_dir_for_callback.clone();
        let device = device_for_callback.clone();
        let overlays = overlays_for_callback.clone();
        let metrics = metrics_for_callback.clone();

        thread::spawn(move || {
            info!("Starting processor for source {} ({})", index, source_display_name(&url));
            if let Err(e) = process_source(
                url.clone(),
                opts,
                &out_dir,
                &data_dir,
                device,
                overlays,
                metrics,
                false,
                Some(stop_signal),
                false,
            ) {
                error!("Source {} processor error: {}", index, e);
            }
            info!("Processor for source {} finished", index);
        })
    });

    // Create upload-only callback (skip MediaMTX, return output path)
    let opts_for_upload = opts.clone();
    let out_dir_for_upload = out_dir.clone();
    let data_dir_for_upload = data_dir.clone();
    let device_for_upload = inference_device.clone();
    let overlays_for_upload = overlays.clone();
    let metrics_for_upload = metrics_map.clone();

    let start_upload_fn: control::StartUploadFn = std::sync::Arc::new(move |url: String, stop_signal: std::sync::Arc<std::sync::atomic::AtomicBool>| {
        let opts = opts_for_upload.clone();
        let out_dir = out_dir_for_upload.clone();
        let data_dir = data_dir_for_upload.clone();
        let device = device_for_upload.clone();
        let overlays = overlays_for_upload.clone();
        let metrics = metrics_for_upload.clone();

        thread::spawn(move || {
            info!("Starting upload processor for {}", source_display_name(&url));
            match process_source(
                url.clone(),
                opts,
                &out_dir,
                &data_dir,
                device,
                overlays,
                metrics,
                false,
                Some(stop_signal),
                true, // skip_mediamtx
            ) {
                Ok(out_path) => {
                    info!("Upload processor finished: {}", out_path);
                    // Return just the filename, not the full path
                    let filename = std::path::Path::new(&out_path)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or(out_path);
                    Some(filename)
                }
                Err(e) => {
                    error!("Upload processor error: {}", e);
                    None
                }
            }
        })
    });

    let jobs: control::SharedJobMap = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));

    info!("Starting control server on 0.0.0.0:9010...");
    let _control_server = control::start_control_server(
        "0.0.0.0:9010",
        overlays.clone(),
        rtsp_config.clone(),
        out_dir.join("analytics.db"),
        source_manager.clone(),
        start_source_fn.clone(),
        start_upload_fn,
        metrics_map.clone(),
        jobs,
    );
    info!("Control server started");

    // Start initially selected sources
    {
        let mut mgr = source_manager.lock().unwrap();
        for source_url in &selected_sources {
            // Find the index for this source
            if let Some(idx) = sources.iter().position(|s| s == source_url) {
                let index = idx + 1;
                let name = source_display_name(source_url);
                let stop_signal = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let handle = {
                    let url = source_url.clone();
                    let stop = stop_signal.clone();
                    Some(start_source_fn(index, url, stop))
                };
                mgr.running.insert(index, RunningSource {
                    index,
                    name,
                    url: source_url.clone(),
                    stop_signal,
                    handle,
                });
            }
        }
        info!("Started {} initial source(s)", mgr.running.len());
    }

    // Main loop - keep running until all sources finish or Ctrl+C
    info!("Backend running. Use API to start/stop sources dynamically.");
    info!("  GET  /api/sources        - list all sources");
    info!("  POST /api/sources/start  - start a source {{\"index\": N}}");
    info!("  POST /api/sources/stop   - stop a source {{\"index\": N}}");
    info!("  POST /api/sources/active - set active sources {{\"indexes\": [1,2,3]}}");
    info!("Press Ctrl+C to shutdown.");

    // Wait for shutdown signal
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        info!("Shutdown signal received...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        thread::sleep(std::time::Duration::from_millis(500));

        // Check if any sources are still running
        let mgr = source_manager.lock().unwrap();
        if mgr.running.is_empty() && !has_source {
            // All sources stopped and not in single-source mode
            // Keep running to allow new sources to be started via API
        }
        drop(mgr);
    }

    // Shutdown: stop all running sources
    info!("Shutting down all sources...");
    {
        let mut mgr = source_manager.lock().unwrap();
        mgr.stop_all();
    }
    info!("Shutdown complete");

    Ok(())
}
