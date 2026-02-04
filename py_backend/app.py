"""Entry point for IRIS Command py_backend.

Sets up multiprocessing queues, starts the relay worker, and launches the FastAPI server.
"""

import os
import time
import threading
import multiprocessing as mp

from log_utils import setup_process_logging
from server import (
    run_control_server,
    update_metrics,
    update_frame,
    update_raw_frame,
    get_overlay_state,
    add_alert,
)
from yolobyte import process_stream, process_upload_stream

# Shared communication objects
spawn_ctx = None
frame_queue = None
raw_frame_queue = None
metrics_queue = None
alert_queue = None
overlay_shared_dict = None


def relay_worker(stop_event, f_q, m_q, a_q, rf_q):
    """Relay metrics, frames, raw frames, and alerts from multiprocess queues to control server."""
    while not stop_event.is_set():
        try:
            for _ in range(60):
                if f_q.empty():
                    break
                name, data = f_q.get_nowait()
                update_frame(name, data)
        except:
            pass

        # Relay raw frames for SAM
        try:
            for _ in range(60):
                if rf_q.empty():
                    break
                name, data = rf_q.get_nowait()
                update_raw_frame(name, data)
        except:
            pass

        try:
            for _ in range(10):
                if m_q.empty():
                    break
                name, data = m_q.get_nowait()
                update_metrics(name, data)
        except:
            pass

        # Process alert queue
        try:
            for _ in range(5):
                if a_q.empty():
                    break
                source, congestion, metrics_data, screenshot = a_q.get_nowait()
                add_alert(source, congestion, metrics_data, screenshot)
        except:
            pass

        time.sleep(0.002)


def start_backend(idx, url, name, overlay_config=None, active_streams=1):
    """Spawn a new inference process for an RTSP source."""
    overlay_shared_dict[name] = overlay_config if overlay_config else get_overlay_state(name)
    stop = spawn_ctx.Event()
    p = spawn_ctx.Process(
        target=process_stream,
        args=(idx, name, url, stop, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_shared_dict, active_streams),
        daemon=True,
    )
    p.start()
    return p, stop


def start_upload_backend(file_path, name, overlay_config=None, is_crowd=False, active_streams=1, realtime=False):
    """Start inference on an uploaded video file."""
    overlay_shared_dict[name] = overlay_config if overlay_config else get_overlay_state(name)
    stop = spawn_ctx.Event()
    p = spawn_ctx.Process(
        target=process_upload_stream,
        args=(name, file_path, stop, frame_queue, metrics_queue, alert_queue, raw_frame_queue, overlay_shared_dict, is_crowd, active_streams, realtime),
        daemon=True,
    )
    p.start()
    return p, stop


def main():
    global spawn_ctx, overlay_shared_dict, frame_queue, raw_frame_queue, metrics_queue, alert_queue
    setup_process_logging("backend")

    # CUDA requires non-fork start method to avoid "Cannot re-initialize CUDA in forked subprocess"
    # Try forkserver first (works with CUDA, fewer SemLock issues), then spawn
    # Allow override via IRIS_MP_START_METHOD=spawn|fork|forkserver.
    available = mp.get_all_start_methods()
    default_method = "forkserver" if "forkserver" in available else "spawn"
    method = os.environ.get("IRIS_MP_START_METHOD", default_method)
    if method not in available:
        method = default_method

    print(f"[MP] Using multiprocessing start method: {method}")
    ctx = mp.get_context(method)
    spawn_ctx = ctx

    manager = ctx.Manager()
    overlay_shared_dict = manager.dict()
    frame_queue = ctx.Queue(maxsize=60)
    raw_frame_queue = ctx.Queue(maxsize=30)
    metrics_queue = ctx.Queue(maxsize=10)
    alert_queue = ctx.Queue(maxsize=10)

    stop_relay = threading.Event()
    relay_t = threading.Thread(target=relay_worker, args=(stop_relay, frame_queue, metrics_queue, alert_queue, raw_frame_queue), daemon=True)
    relay_t.start()

    try:
        run_control_server(start_backend, start_upload_backend, overlay_shared_dict)
    finally:
        stop_relay.set()
        relay_t.join(timeout=1.0)


if __name__ == "__main__":
    main()
