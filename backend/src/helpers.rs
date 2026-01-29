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
        stride: 1,
        headless: false,
        show_heatmap: false,
        show_trails: false,
        show_bboxes: false,
    };

    let mut iter = env::args().skip(1).peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--stride" => {
                let stride = iter.next().ok_or_else(|| anyhow!("--stride requires a value"))?;
                let parsed = stride.parse::<usize>()
                    .map_err(|_| anyhow!("--stride must be a positive integer"))?;
                opts.stride = parsed.max(1);
            }
            _ => { /* unchanged */ }
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
    is_rtsp: bool,
) -> CaptureWorker {
    let (tx, rx) = bounded::<CapturedFrame>(buffer.max(1));
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);

    // For local files, get the source FPS for frame-rate throttling
    let source_fps = if !is_rtsp {
        let fps = cap.get(videoio::CAP_PROP_FPS).unwrap_or(30.0);
        if fps > 0.0 && fps.is_finite() { Some(fps) } else { Some(30.0) }
    } else {
        None
    };

    let join = thread::spawn(move || -> Result<()> {
        let mut frame = Mat::default();
        let mut idx = 0usize;

        // Frame interval for local files (accounting for stride)
        let frame_interval = source_fps.map(|fps| {
            let effective_fps = fps / stride.max(1) as f64;
            std::time::Duration::from_secs_f64(1.0 / effective_fps.max(1.0))
        });
        let mut last_send = std::time::Instant::now();

        loop {
            if stop_thread.load(Ordering::Relaxed) {
                break;
            }

            if !cap.read(&mut frame)? {
                break; // end of file or stream
            }

            if idx % stride != 0 {
                idx += 1;
                continue;
            }

            // Throttle local files to source FPS for stable playback
            if let Some(interval) = frame_interval {
                let elapsed = last_send.elapsed();
                if elapsed < interval {
                    std::thread::sleep(interval - elapsed);
                }
            }

            let owned = frame.try_clone()?;
            let packet = CapturedFrame { frame: owned, idx };

            if is_rtsp {
                let _ = tx.try_send(packet); // drop frames if consumer is slow
            } else {
                let _ = tx.send(packet); // block until consumer is ready
            }

            last_send = std::time::Instant::now();
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
    Full(RtspConfig),
    Map { rtsp_links: Vec<String> },
    List(Vec<String>),
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
    skip_mediamtx: bool,
) -> Result<(SourceState, videoio::VideoCapture)> {
    let label = source_display_name(source);
    let capture_env = "OPENCV_FFMPEG_CAPTURE_OPTIONS";
    let is_rtsp = is_rtsp_source(source);

    let base_opts = if is_rtsp {
        "rtsp_transport;tcp|fflags;discardcorrupt|err_detect;ignore_err|max_delay;300000|reorder_queue_size;1024|stimeout;5000000"
    } else {
        "fflags;discardcorrupt|err_detect;ignore_err"
    };

    if env::var_os(capture_env).is_none() {
        unsafe { env::set_var(capture_env, base_opts); }
    }

    let cap = videoio::VideoCapture::from_file(source, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        return Err(anyhow!(
            "Could not open video source: {}",
            mask_rtsp_credentials(source)
        ));
    }

    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    if fps <= 0.0 || !fps.is_finite() {
        return Err(anyhow!("Invalid FPS: {}", fps));
    }

    let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let stride = stride.max(1);
    let out_fps = (fps / stride as f64).max(1.0);

    let mediamtx = if skip_mediamtx {
        info!("[{}] MediaMTX publishing skipped (file-only mode)", label);
        None
    } else {
        let name = source_display_name(source);
        let publish_url = format!("rtsp://127.0.0.1:8554/processed_{}", name.trim());
        info!("[{}] Publishing to MediaMTX at: {}", label, publish_url);
        Some(start_rtsp_publisher(w, h, out_fps, &publish_url)?)
    };

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let (out_path, codecs, err_label) = match output {
        OutputFormat::Mp4 => (
            out_dir.join(format!("drone_analysis_{timestamp}.mp4")),
            vec![
                videoio::VideoWriter::fourcc('a','v','c','1')?,
                videoio::VideoWriter::fourcc('m','p','4','v')?,
                videoio::VideoWriter::fourcc('H','2','6','4')?,
            ],
            "mp4",
        ),
        OutputFormat::Mkv => (
            out_dir.join(format!("drone_analysis_{timestamp}.mkv")),
            vec![
                videoio::VideoWriter::fourcc('H','2','6','4')?,
                videoio::VideoWriter::fourcc('X','2','6','4')?,
                videoio::VideoWriter::fourcc('M','J','P','G')?,
            ],
            "mkv",
        ),
    };

    let writer = codecs
        .into_iter()
        .find_map(|codec| {
            let wtr = videoio::VideoWriter::new(
                out_path.to_str().unwrap(),
                codec,
                out_fps,
                Size::new(w, h),
                true,
            )
            .ok()?;
            wtr.is_opened().ok().filter(|&ok| ok).map(|_| wtr)
        })
        .ok_or_else(|| anyhow!("VideoWriter failed to open ({err_label})"))?;

    let write_limit = if is_rtsp {
        Some((out_fps * RTSP_SAVE_SECONDS).round().max(1.0) as usize)
    } else {
        None
    };

    let tracker = ByteTracker::new(
        out_fps.round().max(1.0) as usize,
        TRACK_BUFFER,
        TRACK_THRESH,
        HIGH_THRESH,
        MATCH_THRESH,
    );

    let (sys, sys_pid, sys_last_update, sys_snapshot, gpu_util_shared, gpu_poll_stop, gpu_poll_join) =
        init_system_snapshot(&device);

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
            out_path,
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
