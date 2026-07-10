"""
Diagnostic script #5 — the final check: verify that out[0] under model.eval()
(the exact tensor RTDETRPredictor.postprocess() reads as `preds[0]`) has a
real, usable gradient with respect to the noise, using the same
max-confidence-person-row loss now used in train_cloak.py.

Usage:
    python debug_rtdetr_out0_gradient.py --image path/to/one_frame.jpg
"""
import os, argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()
RTDETR_WEIGHTS = os.environ.get('RTDETR_WEIGHTS', os.path.join('pretrained_models', 'rtdetr-l.pt'))

parser = argparse.ArgumentParser()
parser.add_argument('--image',   required=True, type=str)
parser.add_argument('--epsilon', default=8,      type=int)
parser.add_argument('--gpu',     default='0',    type=str)
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'
EPS = args.epsilon / 255.0
_bce = torch.nn.BCELoss()

from ultralytics import YOLO
yolo  = YOLO(RTDETR_WEIGHTS)
model = yolo.model.to(DEVICE).eval()

img = Image.open(args.image).convert('RGB')
base = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
base = F.interpolate(base, size=(640, 640), mode='bilinear', align_corners=False)

torch.manual_seed(0)
noise = torch.empty(3, 640, 640, device=DEVICE).uniform_(-EPS, EPS)
noise.requires_grad_(True)

n_c = torch.clamp(noise, -EPS, EPS)
adv = torch.clamp(base + n_c.unsqueeze(0), 0.0, 1.0)

out = model(adv)
preds = out[0] if isinstance(out, (list, tuple)) else out

print(f"preds shape: {tuple(preds.shape)}  requires_grad={preds.requires_grad}  grad_fn={preds.grad_fn}")

conf     = preds[..., 4]
class_id = preds[..., 5]

print(f"\nTop 5 rows by confidence:")
top_conf, top_idx = conf[0].topk(5)
for c, i in zip(top_conf.tolist(), top_idx.tolist()):
    print(f"  row {i}: conf={c:.4f}  class_id={class_id[0, i].item():.1f}")

person_mask = (class_id < 0.5).float()
person_conf = conf * person_mask
top_person_conf, top_person_idx = person_conf.max(dim=-1)
print(f"\nBest person-class row: idx={top_person_idx.item()}  "
      f"conf={top_person_conf.item():.4f}")

top_person_conf = top_person_conf.clamp(1e-6, 1 - 1e-6)
loss = _bce(top_person_conf, torch.zeros_like(top_person_conf))
print(f"\nloss = {loss.item():.6f}  requires_grad={loss.requires_grad}  grad_fn={loss.grad_fn}")

loss.backward()

if noise.grad is None:
    print("\n❌ noise.grad is None — gradient did not reach the noise tensor.")
else:
    g = noise.grad
    print(f"\n✅ noise.grad exists — abs mean={g.abs().mean().item():.8f}  "
          f"max={g.abs().max().item():.8f}  nonzero={int((g != 0).sum())}/{g.numel()}")

print("\nCompare 'Best person-class row conf' above to what preview_detect.py / "
      "debug_rtdetr_before_after.py reports for this same image via model.predict() — "
      "they should be very close (both read the same out[0] tensor).")