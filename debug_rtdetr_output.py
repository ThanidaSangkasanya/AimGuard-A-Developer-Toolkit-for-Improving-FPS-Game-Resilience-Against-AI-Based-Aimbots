"""
Diagnostic script — run this once to inspect what model(tensor) actually
returns for RT-DETR, so we can fix loss_rtdetr() in train_cloak.py to match
what Ultralytics' model.predict() really decodes at evaluation time.

Usage:
    python debug_rtdetr_output.py --image path/to/one_frame.jpg

Paste the printed output back so the loss function can be corrected.
"""
import os, sys, argparse
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()
RTDETR_WEIGHTS = os.environ.get('RTDETR_WEIGHTS', os.path.join('pretrained_models', 'rtdetr-l.pt'))

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, type=str)
parser.add_argument('--gpu',   default='0',   type=str)
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'

from ultralytics import YOLO
yolo  = YOLO(RTDETR_WEIGHTS)
model = yolo.model.to(DEVICE).eval()

img = Image.open(args.image).convert('RGB')
t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
t = F.interpolate(t, size=(640, 640), mode='bilinear', align_corners=False)

with torch.no_grad():
    out = model(t)

def describe(x, prefix=""):
    if isinstance(x, torch.Tensor):
        print(f"{prefix}Tensor  shape={tuple(x.shape)}  dtype={x.dtype}  "
              f"min={x.min().item():.4f}  max={x.max().item():.4f}")
    elif isinstance(x, (list, tuple)):
        print(f"{prefix}{type(x).__name__} of length {len(x)}:")
        for i, item in enumerate(x):
            describe(item, prefix=prefix + f"  [{i}] ")
    elif isinstance(x, dict):
        print(f"{prefix}dict with keys: {list(x.keys())}")
        for k, v in x.items():
            describe(v, prefix=prefix + f"  ['{k}'] ")
    else:
        print(f"{prefix}{type(x)} -> {x}")

print("\n=== Raw model(tensor) output structure ===")
describe(out)

print("\n=== For comparison: model.predict() result (what eval actually uses) ===")
img_np = np.array(img)
results = yolo.predict(source=img_np, conf=0.25, classes=[0], verbose=False)
print(f"Number of boxes detected: {len(results[0].boxes)}")
for box in results[0].boxes:
    print(f"  xyxy={box.xyxy[0].tolist()}  conf={box.conf[0].item():.3f}  cls={box.cls[0].item()}")

print("\nPlease copy everything above and share it back so loss_rtdetr() can be fixed to match.")