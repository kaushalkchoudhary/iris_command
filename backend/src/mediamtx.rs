use anyhow::Result;
use log::{debug, info, warn, error};
use opencv::core::Mat;
use opencv::prelude::MatTraitConstManual;
use std::{
    io::Write,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, atomic::{AtomicBool, Ordering}},
    thread,
    time::Duration,
};

use crossbeam_channel::{bounded, Sender};

pub struct RtspPublisher {
    tx: Sender<Vec<u8>>,
    alive: Arc<AtomicBool>,
    _child: Child,
}

pub fn start_rtsp_publisher(
    w: i32,
    h: i32,
    fps: f64,
    path: &str,
) -> Result<RtspPublisher> {
    info!("Starting FFmpeg RTSP publisher: {}x{} @ {:.2} FPS -> {}", w, h, fps, path);

    let mut child = Command::new("ffmpeg")
        .args([
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", &format!("{}x{}", w, h),
            "-r", &format!("{:.2}", fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", "30",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdin = child.stdin.take().expect("ffmpeg stdin");

    let (tx, rx) = bounded::<Vec<u8>>(2); // 🔑 SMALL buffer → bounded latency
    let alive = Arc::new(AtomicBool::new(true));
    let alive_thread = alive.clone();

    thread::spawn(move || {
        let mut stdin: ChildStdin = stdin;

        while alive_thread.load(Ordering::Relaxed) {
            match rx.recv() {
                Ok(buf) => {
                    if let Err(e) = stdin.write_all(&buf) {
                        error!("RTSP write failed: {}", e);
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        debug!("RTSP writer thread exiting");
    });

    Ok(RtspPublisher {
        tx,
        alive,
        _child: child,
    })
}

impl RtspPublisher {
    #[inline]
    pub fn send(&mut self, frame: &Mat) -> Result<()> {
        let bytes = frame.data_bytes()?.to_vec();

        // 🔑 Drop frame if encoder is behind
        if self.tx.try_send(bytes).is_err() {
            // silently drop (live > complete)
        }
        Ok(())
    }
}
