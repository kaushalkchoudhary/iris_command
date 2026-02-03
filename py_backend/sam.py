"""SAM3 (Segment Anything Model 3) integration for forensics overlay."""

import sys
import time
import base64
import threading
from pathlib import Path

import numpy as np
import cv2

# ── SAM3 globals ──
SAM_DIR = Path(__file__).resolve().parent.parent / "sam"
sam_lock = threading.Lock()
sam_model = None
sam_processor = None
sam_model_loaded = False

sam_results_lock = threading.Lock()
sam_results = {}  # source -> {annotated_frame, detections, count, timestamp}
sam_threads = {}  # source -> {thread, stop_event, prompt, confidence}

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


def load_sam_model():
    """Lazy-load SAM3 model on CUDA."""
    global sam_model, sam_processor, sam_model_loaded
    if sam_model_loaded:
        return True

    try:
        import torch
        sam_str = str(SAM_DIR)
        if sam_str not in sys.path:
            sys.path.insert(0, sam_str)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = SAM_DIR / "sam3.pt"

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[SAM3] Loading model on CUDA from {checkpoint} ...")
        sam_model = build_sam3_image_model(checkpoint_path=str(checkpoint), device="cuda")
        sam_processor = Sam3Processor(sam_model)
        sam_model_loaded = True
        print("[SAM3] Model loaded successfully on CUDA")
        return True
    except Exception as e:
        print(f"[SAM3] Failed to load model: {e}")
        sam_model = None
        sam_processor = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def sam_annotate_frame(jpeg_data, prompt, confidence=0.7,
                       show_boxes=True, show_masks=True):
    """Run SAM3 inference on a JPEG frame and return annotated image + detections."""
    import torch

    img_array = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return None, []

    h, w = img_array.shape[:2]
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to("cuda")

    with sam_lock:
        sam_processor.confidence_threshold = confidence
        with torch.amp.autocast("cuda", dtype=torch.float16):
            state = sam_processor.set_image(tensor)
            output = sam_processor.set_text_prompt(state=state, prompt=prompt)

    del tensor, state
    torch.cuda.empty_cache()

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

            detections.append({
                "score": round(s, 3),
                "box": [int(bi[0]), int(bi[1]), int(bi[2]), int(bi[3])],
            })

    _, jpeg_out = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_out.tobytes(), detections, w, h


def sam_worker(source, prompt, confidence, stop_event,
               show_boxes=True, show_masks=True, settings_ref=None,
               raw_frame_lock=None, raw_frame_buffer=None):
    """Background thread: grab frame every ~1s, run SAM3, store result."""
    import torch
    print(f"[SAM3] Worker started for {source} with prompt='{prompt}' conf={confidence}")
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
                stop_event.wait(0.5)
                continue

            t0 = time.monotonic()
            annotated_jpeg, detections, fw, fh = sam_annotate_frame(
                jpeg_data, prompt, cur_confidence, cur_show_boxes, cur_show_masks
            )
            if annotated_jpeg is not None:
                b64 = base64.b64encode(annotated_jpeg).decode("ascii")
                with sam_results_lock:
                    sam_results[source] = {
                        "annotated_frame": b64,
                        "detections": detections,
                        "count": len(detections),
                        "prompt": prompt,
                        "confidence": cur_confidence,
                        "show_boxes": cur_show_boxes,
                        "show_masks": cur_show_masks,
                        "timestamp": time.time(),
                        "frame_width": fw,
                        "frame_height": fh,
                    }
            elapsed = time.monotonic() - t0
            sleep_time = max(0.1, 1.0 - elapsed)
        except Exception as e:
            print(f"[SAM3] Worker error for {source}: {e}")
            sleep_time = 1.0

        stop_event.wait(sleep_time)

    # Cleanup GPU memory when worker exits
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with sam_results_lock:
        sam_results.pop(source, None)
    print(f"[SAM3] Worker stopped for {source}")
