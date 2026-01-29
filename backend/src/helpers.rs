use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver};
use chrono::Local;
use jamtrack_rs::byte_tracker::ByteTracker;
use log::{debug, info};
use ndarray::Array3;
use opencv::{
    core::{Mat, Size},
    imgproc,
    prelude::{
        MatExprTraitConst, MatTraitConst, MatTraitConstManual, VideoCaptureTrait,
        VideoCaptureTraitConst, VideoWriterTraitConst,
    },
    videoio,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env, fs,
    path::Path,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Instant,
};
use ultralytics_inference::Device;

use crate::analytics::init_system_snapshot;
use crate::mediamtx::start_rtsp_publisher;
use crate::state::{
    SourceState, RTSP_SAVE_SECONDS, TRACK_BUFFER, TRACK_THRESH, HIGH_THRESH, MATCH_THRESH,
};

#[derive(Clone, Copy, Debug)]
pub enum OutputFormat {
    Mkv,
    Mp4,
}

pub const DEFAULT_IO_BUFFER: usize = 120;

#[derive(Debug, Clone)]
pub struct CliOptions {
    pub source: Option<String>,
    pub use_rtsp: bool,
    pub rtsp_indexes: Vec<usize>,
    pub output: OutputFormat,
    pub device: Option<Device>,
    pub threads: Option<usize>,
    pub io_buffer: usize,
    pub stride: usize,
    pub headless: bool,
    pub show_heatmap: bool,
    pub show_trails: bool,
    pub show_bboxes: bool,
}

pub fn parse_args() -> Result<CliOptions> {
    let mut opts = CliOptions {
        source: None,
        use_rtsp: false,
        rtsp_indexes: Vec::new(),
        output: OutputFormat::Mkv,
        device: None,
        threads: None,
        io_buffer: DEFAULT_IO_BUFFER,
        stride: 2,
        headless: false,
        show_heatmap: false,
        show_trails: false,
        show_bboxes: false,
    };

    let mut iter = env::args().skip(1).peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--rtsp" => opts.use_rtsp = true,
            "--mp4" => opts.output = OutputFormat::Mp4,
            "--mkv" => opts.output = OutputFormat::Mkv,
            "--index" => {
                let idx = iter.next().ok_or_else(|| anyhow!("--index requires a value"))?;
                let parsed = idx
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--index must be a positive integer"))?;
                opts.rtsp_indexes.push(parsed.saturating_sub(1));
            }
            "--indexes" => {
                let raw = iter.next().ok_or_else(|| anyhow!("--indexes requires a value"))?;
                for part in raw.split(',') {
                    let trimmed = part.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let parsed = trimmed
                        .parse::<usize>()
                        .map_err(|_| anyhow!("--indexes must be a comma-separated list of positive integers"))?;
                    opts.rtsp_indexes.push(parsed.saturating_sub(1));
                }
            }
            "--threads" => {
                let threads = iter.next().ok_or_else(|| anyhow!("--threads requires a value"))?;
                let parsed = threads
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--threads must be a non-negative integer"))?;
                opts.threads = Some(parsed);
            }
            "--stride" => {
                let stride = iter.next().ok_or_else(|| anyhow!("--stride requires a value"))?;
                let parsed = stride
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--stride must be a positive integer"))?;
                opts.stride = parsed.max(1);
            }
            "--headless" => {
                opts.headless = true;
            }
            "--buffer" | "--io-buffer" => {
                let buf = iter.next().ok_or_else(|| anyhow!("--buffer requires a value"))?;
                let parsed = buf
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--buffer must be a non-negative integer"))?;
                opts.io_buffer = parsed.max(1);
            }
            "--device" => {
                let device = iter.next().ok_or_else(|| anyhow!("--device requires a value"))?;
                let parsed = Device::from_str(&device)
                    .map_err(|err| anyhow!("Invalid --device value '{device}': {err}"))?;
                opts.device = Some(parsed);
            }
            "--cpu" => {
                opts.device = Some(Device::Cpu);
            }
            "--gpu" => {
                let device = if cfg!(target_os = "macos") {
                    Device::Mps
                } else {
                    Device::Cuda(0)
                };
                opts.device = Some(device);
            }
            "--heatmap" => {
                opts.show_heatmap = true;
            }
            "--trails" => {
                opts.show_trails = true;
            }
            "--bbox" | "--bboxes" => {
                opts.show_bboxes = true;
            }
            _ if arg.starts_with("--") => {
                let rest = arg.trim_start_matches("--");
                if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                    let parsed = rest
                        .parse::<usize>()
                        .map_err(|_| anyhow!("Invalid RTSP index flag: {arg}"))?;
                    opts.rtsp_indexes.push(parsed.saturating_sub(1));
                } else if opts.source.is_none() {
                    opts.source = Some(arg);
                }
            }
            _ => {
                if opts.source.is_none() {
                    opts.source = Some(arg);
                }
            }
        }
    }

    Ok(opts)
}

pub struct CaptureWorker {
    pub rx: Receiver<CapturedFrame>,
    stop: Arc<AtomicBool>,
    join: JoinHandle<Result<()>>,
}

impl CaptureWorker {
    pub fn stop(self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        match self.join.join() {
            Ok(res) => res,
            Err(_) => Err(anyhow!("Capture thread panicked")),
        }
    }
}

pub struct CapturedFrame {
    pub frame: Mat,
    pub idx: usize,
}

pub fn start_capture_thread(
    mut cap: videoio::VideoCapture,
    stride: usize,
    buffer: usize,
) -> CaptureWorker {
    let (tx, rx) = bounded::<CapturedFrame>(buffer.max(1));
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let join = thread::spawn(move || -> Result<()> {
        let mut frame = Mat::default();
        let mut idx = 0usize;
        loop {
            if stop_thread.load(Ordering::Relaxed) {
                break;
            }
            if !cap.read(&mut frame)? {
                break;
            }
            if idx % stride != 0 {
                idx += 1;
                continue;
            }
            let owned = frame.try_clone()?;
            let packet = CapturedFrame { frame: owned, idx };
            if tx.try_send(packet).is_err() {
                // Drop newest frame when the buffer is full to avoid blocking I/O.
            }
            idx += 1;
        }
        Ok(())
    });

    CaptureWorker { rx, stop, join }
}

pub fn mat_to_array3_rgb(mat: &Mat) -> Result<Array3<u8>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        mat,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
    )?;

    let rows = rgb.rows() as usize;
    let cols = rgb.cols() as usize;

    let data = rgb.data_bytes()?.to_vec();
    Ok(Array3::from_shape_vec((rows, cols, 3), data)?)
}

/// Overlay configuration for a single stream
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OverlayConfig {
    #[serde(default)]
    pub heatmap: bool,
    #[serde(default)]
    pub trails: bool,
    #[serde(default)]
    pub bboxes: bool,
}

/// Full RTSP configuration with overlay settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RtspConfig {
    #[serde(default)]
    pub rtsp_links: Vec<String>,
    /// 1-based indexes of active RTSP sources (empty = use first available)
    #[serde(default)]
    pub active_sources: Vec<usize>,
    #[serde(default)]
    pub overlays: HashMap<String, OverlayConfig>,
}

// Legacy format support
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RtspYaml {
    List(Vec<String>),
    Map { rtsp_links: Vec<String> },
    Full(RtspConfig),
}

pub fn load_rtsp_config(path: &Path) -> Result<RtspConfig> {
    if !path.exists() {
        debug!("RTSP config file not found: {}", path.display());
        return Ok(RtspConfig::default());
    }
    debug!("Reading RTSP config: {}", path.display());
    let raw = fs::read_to_string(path)?;
    let parsed: RtspYaml = serde_yaml::from_str(&raw)?;
    let config = match parsed {
        RtspYaml::List(list) => RtspConfig {
            rtsp_links: list,
            active_sources: Vec::new(),
            overlays: HashMap::new(),
        },
        RtspYaml::Map { rtsp_links } => RtspConfig {
            rtsp_links,
            active_sources: Vec::new(),
            overlays: HashMap::new(),
        },
        RtspYaml::Full(config) => config,
    };
    debug!("Parsed {} RTSP link(s) and {} overlay config(s)",
           config.rtsp_links.len(), config.overlays.len());
    Ok(config)
}

pub fn load_rtsp_links(path: &Path) -> Result<Vec<String>> {
    Ok(load_rtsp_config(path)?.rtsp_links)
}

pub fn save_rtsp_config(path: &Path, config: &RtspConfig) -> Result<()> {
    debug!("Saving RTSP config to: {}", path.display());
    let yaml = serde_yaml::to_string(config)?;
    fs::write(path, yaml)?;
    info!("RTSP config saved to {}", path.display());
    Ok(())
}

pub fn is_rtsp_source(source: &str) -> bool {
    let s = source.trim();
    s.starts_with("rtsp://") || s.starts_with("rtsps://")
}

/// Mask credentials in RTSP URLs for safe logging
/// e.g., rtsp://user:password@host:554/stream -> rtsp://user:****@host:554/stream
pub fn mask_rtsp_credentials(url: &str) -> String {
    let trimmed = url.trim();
    if !is_rtsp_source(trimmed) {
        return trimmed.to_string();
    }

    // Find the @ symbol which separates credentials from host
    if let Some(at_pos) = trimmed.find('@') {
        // Find the :// prefix end
        if let Some(scheme_end) = trimmed.find("://") {
            let after_scheme = scheme_end + 3;
            let creds_part = &trimmed[after_scheme..at_pos];
            // Check if there's a password (colon in credentials)
            if let Some(colon_pos) = creds_part.find(':') {
                let user = &creds_part[..colon_pos];
                let scheme = &trimmed[..after_scheme];
                let rest = &trimmed[at_pos..];
                return format!("{}{}:****{}", scheme, user, rest);
            }
        }
    }
    trimmed.to_string()
}

pub fn source_display_name(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if is_rtsp_source(trimmed) {
        let base = trimmed.split('?').next().unwrap_or(trimmed);
        let last = base.rsplit('/').next().unwrap_or(base);
        return last.to_string();
    }

    let base = trimmed.split('?').next().unwrap_or(trimmed);
    Path::new(base)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(base)
        .to_string()
}

pub fn init_source_state(
    source: &str,
    out_dir: &Path,
    output: OutputFormat,
    device: Device,
    stride: usize,
) -> Result<(SourceState, videoio::VideoCapture)> {
    let label = source_display_name(source);
    let capture_env = "OPENCV_FFMPEG_CAPTURE_OPTIONS";
    let capture_opts_set = env::var_os(capture_env).is_some();
    let is_rtsp = is_rtsp_source(source);

    debug!("[{}] Configuring FFmpeg capture options...", label);
    let base_opts = if is_rtsp {
        debug!("[{}] Using RTSP transport options (TCP, 5s timeout)", label);
        "rtsp_transport;tcp|fflags;discardcorrupt|err_detect;ignore_err|max_delay;500000|reorder_queue_size;0|stimeout;5000000"
    } else {
        debug!("[{}] Using file capture options", label);
        "fflags;discardcorrupt|err_detect;ignore_err"
    };

    let mut cap = None;
    if cap.is_none() {
        if !capture_opts_set {
            debug!("[{}] Setting {}", label, capture_env);
            unsafe {
                env::set_var(capture_env, base_opts);
            }
        }
        debug!("[{}] Opening video capture with FFmpeg backend...", label);
        cap = Some(videoio::VideoCapture::from_file(source, videoio::CAP_FFMPEG)?);
    }
    let cap = cap.expect("capture is set");
    if !cap.is_opened()? {
        return Err(anyhow!("Could not open video source: {}", mask_rtsp_credentials(source)));
    }
    debug!("[{}] Video capture opened successfully", label);

    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    if fps <= 0.0 {
        return Err(anyhow!("Invalid FPS: {}", fps));
    }

    let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    debug!("[{}] Video stream info: {}x{} @ {:.2} FPS", label, w, h, fps);

    let force_stream = std::env::var("ALWAYS_STREAM")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false);
    let mediamtx = if is_rtsp_source(source) || force_stream {
        let publish_url = std::env::var("MEDIAMTX_PUBLISH_URL").unwrap_or_else(|_| {
            let stream_name = if is_rtsp_source(source) {
                source_display_name(source)
            } else {
                "analytics".to_string()
            };
            let stream_name = if stream_name.trim().is_empty() {
                "analytics"
            } else {
                stream_name.trim()
            };
            format!("rtsp://127.0.0.1:8554/processed_{stream_name}")
        });
        info!("[{}] Starting RTSP republisher to: {}", label, publish_url);
        Some(start_rtsp_publisher(w, h, fps, &publish_url)?)
    } else {
        debug!("[{}] RTSP republishing disabled (local file source)", label);
        None
    };

    let stride = stride.max(1);
    let out_fps = (fps / stride as f64).max(1.0);
    debug!("[{}] Output FPS: {:.2} (stride {})", label, out_fps, stride);

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    debug!("[{}] Setting up video writer...", label);
    let (out_path, codecs, err_label) = match output {
        OutputFormat::Mp4 => (
            out_dir.join(format!("drone_analysis_{timestamp}.mp4")),
            vec![
                videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?,
                videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?,
                videoio::VideoWriter::fourcc('H', '2', '6', '4')?,
                videoio::VideoWriter::fourcc('X', '2', '6', '4')?,
            ],
            "mp4",
        ),
        OutputFormat::Mkv => (
            out_dir.join(format!("drone_analysis_{timestamp}.mkv")),
            vec![
                videoio::VideoWriter::fourcc('H', '2', '6', '4')?,
                videoio::VideoWriter::fourcc('X', '2', '6', '4')?,
                videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?,
                videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?,
            ],
            "mkv",
        ),
    };

    let open_writer = || -> Result<videoio::VideoWriter> {
        for codec in &codecs {
            let candidate = match videoio::VideoWriter::new(
                out_path.to_str().unwrap(),
                *codec,
                out_fps,
                Size::new(w, h),
                true,
            ) {
                Ok(candidate) => candidate,
                Err(_) => continue,
            };
            if candidate.is_opened()? {
                return Ok(candidate);
            }
        }
        Err(anyhow!("VideoWriter failed to open ({err_label})."))
    };

    let writer = open_writer()?;
    info!("[{}] Video writer opened: {}", label, out_path.display());

    let write_limit = if is_rtsp_source(source) {
        let limit = (out_fps * RTSP_SAVE_SECONDS).round().max(1.0) as usize;
        debug!("[{}] RTSP recording limit: {} frames ({:.0}s)", label, limit, RTSP_SAVE_SECONDS);
        Some(limit)
    } else {
        None
    };

    debug!("[{}] Initializing ByteTracker (buffer={}, thresh={:.2}, high={:.2}, match={:.2})",
           label, TRACK_BUFFER, TRACK_THRESH, HIGH_THRESH, MATCH_THRESH);
    let tracker = ByteTracker::new(
        out_fps.round().max(1.0) as usize,
        TRACK_BUFFER,
        TRACK_THRESH,
        HIGH_THRESH,
        MATCH_THRESH,
    );

    debug!("[{}] Initializing system monitoring...", label);
    let (sys, sys_pid, sys_last_update, sys_snapshot, gpu_util_shared, gpu_poll_stop, gpu_poll_join) =
        init_system_snapshot(&device);
    debug!("[{}] Source state initialization complete", label);

    Ok((
        SourceState {
            fps,
            w,
            h,
            stride,
            out_fps,
            writer,
            writer_active: true,
            write_limit,
            frames_written: 0,
            tracker,
            tracks: HashMap::new(),
            next_local_id: -1,
            heatmap: Mat::zeros(h, w, opencv::core::CV_32F)?.to_mat()?,
            source_label: source_display_name(source),
            sys,
            sys_pid,
            sys_last_update,
            sys_snapshot,
            gpu_util_shared,
            gpu_poll_stop,
            gpu_poll_join,
            fps_last_update: Instant::now(),
            fps_frames: 0,
            fps_value: 0.0,
            traffic_density_ema: 0.0,
            mobility_index_ema: 0.0,
            mediamtx,
            device,
        },
        cap,
    ))
}

pub fn draw_source_label(frame: &mut Mat, label: &str) -> Result<()> {
    let title = if label.is_empty() { "source" } else { label };
    let scale = 0.7;
    let thickness = 2;
    imgproc::put_text(
        frame,
        title,
        opencv::core::Point::new(30, 30),
        imgproc::FONT_HERSHEY_DUPLEX,
        scale,
        opencv::core::Scalar::all(255.0),
        thickness,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

pub fn ema_update(prev: &mut f32, value: f32, alpha: f32) -> f32 {
    if *prev == 0.0 {
        *prev = value;
    } else {
        *prev = alpha * value + (1.0 - alpha) * *prev;
    }
    *prev
}
