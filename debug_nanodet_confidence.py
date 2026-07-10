"""
Diagnostic script — NanoDet version of the RT-DETR checks we did earlier.

Compares:
  1. The "confidence" our loss_nanodet() computes (sigmoid of raw head
     output channel 0, from a manual backbone->fpn->head forward pass)
  2. The REAL confidence model.inference() reports for the same image
     (the same call get_cloak.py's detect_nanodet() uses)

If these two numbers are wildly different, our loss is targeting the
wrong thing (same root cause as the RT-DETR bug). If they're close,
the loss is fine and the lower NanoDet DSR is coming from elsewhere
(e.g. signal dilution across many grid cells, similar to what we found
before the RT-DETR max-query fix).

Usage:
    python debug_nanodet_confidence.py --image path/to/one_frame.jpg
"""
import os, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
NANODET_ROOT = os.environ.get('NANODET_ROOT', os.path.join(os.path.dirname(__file__), 'third_party', 'nanodet'))

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, type=str)
parser.add_argument('--conf',  default=0.4,    type=float)
parser.add_argument('--gpu',   default='0',    type=str)
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'
_bce = nn.BCELoss()

sys.path.insert(0, NANODET_ROOT)
from nanodet.util import cfg, load_config, Logger as NL
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
from nanodet.data.collate import naive_collate
from nanodet.data.batch_process import stack_batch_img

load_config(cfg, os.path.join(NANODET_ROOT, 'config', 'nanodet-plus-m_320.yml'))
print(f"Model classes (cfg): {cfg.class_names if hasattr(cfg, 'class_names') else 'N/A'}")
print(f"num_classes in head cfg: {cfg.model.arch.head.num_classes}")

model = build_model(cfg.model)
ckpt  = torch.load(os.path.join(NANODET_ROOT, 'nanodet', 'nanodet-plus-m_320.pth'), map_location='cpu')
load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
model = model.to(DEVICE).eval()
pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
print("Model loaded.\n")

img_pil = Image.open(args.image).convert('RGB')
img_np  = np.array(img_pil)

# ── Path A: the REAL inference path (same as detect_nanodet in get_cloak.py) ──
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
h, w = img_bgr.shape[:2]
meta = dict(img_info={'id': 0, 'file_name': None, 'height': h, 'width': w}, raw_img=img_bgr, img=img_bgr)
meta = pipeline(None, meta, cfg.data.val.input_size)
meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).to(DEVICE)
meta = naive_collate([meta])
meta['img'] = stack_batch_img(meta['img'], divisible=32)
with torch.no_grad():
    results = model.inference(meta)
real_boxes = results[0].get(0, [])
print(f"=== Path A: model.inference() (REAL, what detect_nanodet uses) ===")
if real_boxes:
    real_boxes_sorted = sorted(real_boxes, key=lambda d: d[-1], reverse=True)
    for d in real_boxes_sorted[:5]:
        print(f"  box conf={d[-1]:.4f}")
    top_real_conf = real_boxes_sorted[0][-1]
else:
    print("  No class-0 boxes returned at all (empty).")
    top_real_conf = None

# ── Path B: the loss_nanodet() manual forward path ────────────────────────
base = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
base = F.interpolate(base, size=(320, 320), mode='bilinear', align_corners=False)

noise = torch.zeros(3, 320, 320, device=DEVICE, requires_grad=True)
adv = torch.clamp(base + noise.unsqueeze(0), 0.0, 1.0)

mean = torch.tensor([103.53, 116.28, 123.675], device=DEVICE).view(1, 3, 1, 1) / 255.0
std  = torch.tensor([57.375, 57.12, 58.395], device=DEVICE).view(1, 3, 1, 1) / 255.0
img_norm = (adv - mean) / std
feats = model.backbone(img_norm)
feats = model.fpn(feats)
preds = model.head(feats)

print(f"\n=== Path B: manual forward (what loss_nanodet computes) ===")
print(f"Number of FPN levels returned by head: {len(preds)}")
all_max = []
for i, p in enumerate(preds):
    if isinstance(p, (list, tuple)):
        p = p[0]
    print(f"  level {i}: shape={tuple(p.shape)}  requires_grad={p.requires_grad}")
    prob = torch.sigmoid(p[..., 0:1])
    all_max.append(prob.max().item())
    print(f"    channel-0 sigmoid: max={prob.max().item():.4f}  mean={prob.mean().item():.6f}")

top_manual_conf = max(all_max)
print(f"\nTop confidence via manual forward (channel 0, our loss target): {top_manual_conf:.4f}")
if top_real_conf is not None:
    print(f"Top confidence via model.inference() (REAL):                    {top_real_conf:.4f}")
    print(f"Difference: {abs(top_manual_conf - top_real_conf):.4f}")
    if abs(top_manual_conf - top_real_conf) > 0.15:
        print("\n⚠️  LARGE MISMATCH — loss_nanodet is likely NOT targeting the same value")
        print("    model.inference() actually reports. Same class of bug as RT-DETR.")
    else:
        print("\n✅ Close match — loss_nanodet's target tensor appears correct.")
        print("   The DSR gap is more likely signal dilution across many grid cells")
        print("   (same fix category as RT-DETR's max-query targeting).")

# ── Gradient check ─────────────────────────────────────────────────────────
total = torch.tensor(0.0, device=DEVICE)
for p in preds:
    if isinstance(p, (list, tuple)):
        p = p[0]
    prob = torch.sigmoid(p[..., 0:1])
    total = total + _bce(prob, torch.zeros_like(prob))
total.backward()
if noise.grad is None:
    print("\n❌ noise.grad is None — no gradient reached the noise tensor.")
else:
    g = noise.grad
    print(f"\n✅ noise.grad exists — abs mean={g.abs().mean().item():.8f}  "
          f"max={g.abs().max().item():.8f}  nonzero={int((g != 0).sum())}/{g.numel()}")