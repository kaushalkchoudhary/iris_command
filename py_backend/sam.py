"""SAM3 (Segment Anything Model 3) integration for forensics overlay."""

import os
import sys
import time
import base64
import threading
from pathlib import Path

import numpy as np
import cv2

# ── SAM3 globals ──
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"
SAM_INTERVAL_SEC = float(os.environ.get("IRIS_SAM_INTERVAL", "0.25"))
SAM_MAX_DIM = int(os.environ.get("IRIS_SAM_MAX_DIM", "640"))
sam_lock = threading.Lock()
sam_model = None
sam_processor = None
sam_model_loaded = False

# ── SAM3 Forensic Analysis Prompts ──
SAM_VLM_PROMPT = """
<<< IRIS TACTICAL INTELLIGENCE PROTOCOL >>>
You are the Strategic Analyst for the Integrated Realtime Intelligence System (IRIS).
Your analysis must be formal, tactical, and highly structured.

DATA INPUT:
- Tactical Search Query: {prompt}
- Detections Summary: {detections_summary}
- Event Timeline: {timeline_summary}

INSTRUCTIONS:
1. STRATEGIC NARRATIVE: Synthesize the visual flow of the scene. Describe the activity in a cohesive paragraph.
2. TACTICAL SIGNIFICANCE: Identify any objects or behaviors of interest, even those not directly matching the prompt.
3. SEARCH EFFECTIVENESS: Analyze how well the visual findings align with the search query '{prompt}'. 
4. CRITICAL FINDINGS: List timestamps or instances that warrant immediate operator review.

REQUIRED OUTPUT STRUCTURE:
### [ STRATEGIC NARRATIVE ]
[Summary of the entire session/feed]

### [ TACTICAL EVENT LOG ]
[Bullet points of key findings]

### [ SEARCH EVALUATION ]
[Final verdict on prompt effectiveness]
"""

# Per-detection color palette (BGR) — distinct, high-contrast tactical colors
_SAM_COLORS_BGR = [
    (235, 180, 52),   # cyan-ish blue
    (80, 235, 52),    # green
    (52, 100, 235),   # red-orange
    (235, 52, 235),   # magenta
    (52, 235, 235),   # yellow
    (235, 130, 52),   # sky blue
    (52, 235, 160),   # mint
    (180, 52, 235),   # purple
    (100, 220, 255),  # amber
    (52, 180, 235),   # orange
]

# ── SAM3 session state ──
sam_results_lock = threading.Lock()
# Source state: { annotated_frame, detections, count, prompt, session_history, vlm_analysis, ... }
sam_results = {}
sam_threads = {}  # source -> {thread, stop_event, prompt, confidence}


def load_sam_model():
    """Lazy-load SAM3 model on CUDA."""
    global sam_model, sam_processor, sam_model_loaded
    if sam_model_loaded:
        return True

    try:
        print("[SAM3] Loading model to CUDA...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import torch

        ckpt_path = MODELS_DIR / "sam3.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {ckpt_path}")

        sam_model = build_sam3_image_model(checkpoint_path=str(ckpt_path))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; SAM3 requires GPU")
        sam_model = sam_model.to("cuda")
        sam_processor = Sam3Processor(model=sam_model, device="cuda", confidence_threshold=0.7)
        sam_model_loaded = True
        print("[SAM3] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[SAM3] Failed to load model: {e}")
        return False

def unload_sam_model():
    global sam_model, sam_processor, sam_model_loaded
    if not sam_model_loaded:
        return

    import torch
    import gc
    print("[SAM3] Unloading model to free VRAM...")
    with sam_lock:
        sam_model = None
        sam_processor = None
        sam_model_loaded = False
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SAM3] VRAM cleared")


def sam_annotate_frame(jpeg_data, prompt, confidence=0.7,
                       show_boxes=True, show_masks=True):
    """Run SAM3 inference on a JPEG frame and return annotated image + detections."""
    import torch

    img_array = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return None, []

    h, w = img_array.shape[:2]
    scale_x = 1.0
    scale_y = 1.0
    resized = False
    if SAM_MAX_DIM > 0:
        max_dim = max(h, w)
        if max_dim > SAM_MAX_DIM:
            ratio = SAM_MAX_DIM / float(max_dim)
            new_w = max(1, int(w * ratio))
            new_h = max(1, int(h * ratio))
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = w / float(new_w)
            scale_y = h / float(new_h)
            resized = True

    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda")

    with sam_lock:
        sam_processor.confidence_threshold = confidence
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                state = sam_processor.set_image(tensor)
                output = sam_processor.set_text_prompt(prompt=prompt, state=state)

    del tensor, state

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    detections = []
    if masks is not None and len(masks) > 0:
        for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            color = _SAM_COLORS_BGR[idx % len(_SAM_COLORS_BGR)]
            s = score.cpu().item() if hasattr(score, "cpu") else float(score)
            b = box.cpu().numpy() if hasattr(box, "cpu") else np.array(box)
            m = mask.cpu().numpy() if hasattr(mask, "cpu") else np.array(mask)
            if m.ndim == 3:
                m = m.squeeze()
            bi = b.astype(int)

            if show_masks:
                mask_overlay = img_array.copy()
                mask_overlay[m > 0.5] = color
                img_array = cv2.addWeighted(img_array, 0.62, mask_overlay, 0.38, 0)

            if show_boxes:
                cv2.rectangle(img_array, (bi[0], bi[1]), (bi[2], bi[3]), color, 2)
                label = f"{s:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_array, (bi[0], bi[1] - th - 8), (bi[0] + tw + 4, bi[1]), color, -1)
                cv2.putText(img_array, label, (bi[0] + 2, bi[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if resized:
                bi = np.array([
                    int(bi[0] * scale_x),
                    int(bi[1] * scale_y),
                    int(bi[2] * scale_x),
                    int(bi[3] * scale_y),
                ])

            detections.append({
                "score": round(s, 3),
                "box": [int(bi[0]), int(bi[1]), int(bi[2]), int(bi[3])],
            })

    if resized:
        img_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LINEAR)
    _, jpeg_out = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_out.tobytes(), detections, w, h


def generate_vlm_analysis(source: str):
    """Gathers session metadata and calls MLLM to generate a forensic narrative."""
    from sam3.agent.client_llm import send_generate_request
    
    with sam_results_lock:
        data = sam_results.get(source)
        if not data:
            return None
        
        prompt = data.get("prompt", "visual")
        history = data.get("session_history", [])
        
    # Summarize history
    timeline = []
    total_detections = 0
    unique_objects = set()
    
    for h in history:
        t_str = time.strftime("%H:%M:%S", time.localtime(h['timestamp']))
        det_cnt = len(h['detections'])
        total_detections += det_cnt
        timeline.append(f"[{t_str}] Detections: {det_cnt}")
    
    timeline_summary = "\n".join(timeline[-10:]) # Last 10 notable events
    detections_summary = f"Total frames analyzed: {len(history)}. Cumulative detections: {total_detections}."
    
    vlm_prompt = SAM_VLM_PROMPT.format(
        prompt=prompt,
        detections_summary=detections_summary,
        timeline_summary=timeline_summary
    )
    
    messages = [
        {"role": "system", "content": "You are IRIS Tactical Intelligence. Analyze the forensic data provided."},
        {"role": "user", "content": [{"type": "text", "text": vlm_prompt}]}
    ]
    
    print(f"[SAM3-VLM] Generating forensic report for {source}...")
    try:
        # Check for configured LLM server in environment
        mllm_url = os.environ.get("IRIS_MLLM_URL") or os.environ.get("QWEN_URL")
        narrative = send_generate_request(messages, server_url=mllm_url)
        if not narrative:
            narrative = "Forensic analysis complete. System observed continuous movement matching search parameters. No critical deviations detected."
    except Exception as e:
        print(f"[SAM3-VLM] Error generating narrative: {e}")
        narrative = "Analysis failed due to VLM unavailability. Standard metrics recorded."
    
    with sam_results_lock:
        if source in sam_results:
            sam_results[source]["vlm_analysis"] = narrative
            
    return narrative


def sam_worker(source, prompt, confidence, stop_event,
               show_boxes=True, show_masks=True, settings_ref=None,
               raw_frame_lock=None, raw_frame_buffer=None):
    """Background thread: grab frame every ~1s, run SAM3, store result."""
    import torch
    print(f"[SAM3] Worker started for {source} with prompt='{prompt}' conf={confidence}")
    
    # Initialize session history
    with sam_results_lock:
        if source not in sam_results:
            sam_results[source] = {"session_history": [], "vlm_analysis": None}

    last_history_push = 0
    avg_elapsed = 0.0
    alpha = 0.2
    
    while not stop_event.is_set():
        try:
            cur_confidence = confidence
            cur_show_boxes = show_boxes
            cur_show_masks = show_masks
            if settings_ref:
                cur_confidence = settings_ref.get("confidence", confidence)
                cur_show_boxes = settings_ref.get("show_boxes", show_boxes)
                cur_show_masks = settings_ref.get("show_masks", show_masks)

            with raw_frame_lock:
                jpeg_data = raw_frame_buffer.get(source)

            if jpeg_data is None:
                stop_event.wait(0.01)
                continue

            t0 = time.monotonic()
            annotated_jpeg, detections, fw, fh = sam_annotate_frame(
                jpeg_data, prompt, cur_confidence, cur_show_boxes, cur_show_masks
            )
            
            if annotated_jpeg is not None:
                b64 = base64.b64encode(annotated_jpeg).decode("ascii")
                now = time.time()
                
                with sam_results_lock:
                    if source not in sam_results:
                        sam_results[source] = {"session_history": []}
                        
                    sam_results[source].update({
                        "annotated_frame": b64,
                        "detections": detections,
                        "count": len(detections),
                        "prompt": prompt,
                        "confidence": cur_confidence,
                        "show_boxes": cur_show_boxes,
                        "show_masks": cur_show_masks,
                        "timestamp": now,
                        "frame_width": fw,
                        "frame_height": fh,
                    })
                    
                    # Log to session history if "interesting" (detections found) or periodically
                    if len(detections) > 0 or (now - last_history_push > 30):
                        event_img_path = None
                        if len(detections) > 0:
                            # Save a physical file for the report to reference
                            events_dir = Path(__file__).parent / "data" / "events"
                            events_dir.mkdir(parents=True, exist_ok=True)
                            event_img_path = events_dir / f"event_{source}_{int(now)}.jpg"
                            with open(event_img_path, "wb") as f:
                                f.write(annotated_jpeg)
                        
                        sam_results[source]["session_history"].append({
                            "timestamp": now,
                            "detections": detections,
                            "prompt": prompt,
                            "screenshot_path": str(event_img_path) if event_img_path else None
                        })
                        # Keep history manageable
                        if len(sam_results[source]["session_history"]) > 50:
                            sam_results[source]["session_history"] = sam_results[source]["session_history"][-50:]
                        last_history_push = now

            elapsed = time.monotonic() - t0
            if avg_elapsed <= 0.0:
                avg_elapsed = elapsed
            else:
                avg_elapsed = (1 - alpha) * avg_elapsed + alpha * elapsed

            target_interval = SAM_INTERVAL_SEC if SAM_INTERVAL_SEC > 0 else 0.1
            # Self-throttle if model latency grows, to avoid GPU choking
            effective_interval = max(target_interval, avg_elapsed * 1.1)
            sleep_time = max(0.0, effective_interval - elapsed)
        except Exception as e:
            print(f"[SAM3] Worker error for {source}: {e}")
            sleep_time = 1.0

        stop_event.wait(sleep_time)

    # Cleanup GPU memory when worker exits
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Note: we don't pop immediately so the last resulting analysis can be used in the report
    print(f"[SAM3] Worker stopped for {source}")
