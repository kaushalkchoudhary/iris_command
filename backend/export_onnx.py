from ultralytics import YOLO
import sys

try:
    print("Loading model...")
    model = YOLO('data/best_head.pt')
    
    print("Exporting to ONNX...")
    # Export with dynamic=True to allow different batch sizes if needed, or static if preferred.
    # For trtexec, usually static is simpler, but dynamic is more flexible.
    # Let's do dynamic=False (default usually) or specify checks.
    # Ultralytics export to onnx typically creates a well-formed onnx.
    # simplify=True is good.
    path = model.export(format='onnx', simplify=True)
    print(f"Exported to: {path}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
