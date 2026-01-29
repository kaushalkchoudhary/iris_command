use anyhow::Result;
use log::{debug, info, warn};
use opencv::core::Mat;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread;
use std::time::Duration;
use opencv::prelude::MatTraitConstManual;

pub struct RtspPublisher {
    _child: Child,
    stdin: ChildStdin,
}

pub fn start_rtsp_publisher(
    w: i32,
    h: i32,
    fps: f64,
    path: &str,
) -> Result<RtspPublisher> {
    info!("Starting FFmpeg RTSP publisher: {}x{} @ {:.2} FPS -> {}", w, h, fps, path);

    let spawn_ffmpeg = || -> Result<Child> {
        debug!("Spawning FFmpeg process with libx264 encoder...");
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", &format!("{}x{}", w, h),
            "-r", &format!("{:.2}", fps),
            "-i", "-",
            "-an",
        ]);
        cmd.args([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
        ]);
        cmd.args([
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            path,
        ]);
        Ok(cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?)
    };

    // Force libx264 for compatibility
    let mut child = spawn_ffmpeg()?;
    thread::sleep(Duration::from_millis(120));
    if let Ok(Some(_)) = child.try_wait() {
         warn!("FFmpeg exited early, retrying...");
         child = spawn_ffmpeg()?;
    }

    let stdin = child.stdin.take().expect("ffmpeg stdin");
    info!("FFmpeg RTSP publisher started successfully");

    Ok(RtspPublisher {
        _child: child,
        stdin,
    })
}

impl RtspPublisher {
    #[inline]
    pub fn send(&mut self, frame: &Mat) -> Result<()> {
        self.stdin.write_all(frame.data_bytes()?)?;
        Ok(())
    }
}
