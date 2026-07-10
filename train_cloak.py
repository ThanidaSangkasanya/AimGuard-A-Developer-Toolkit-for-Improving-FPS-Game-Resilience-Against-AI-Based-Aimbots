import os, sys, time, argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Force UTF-8 output so unicode chars (arrows) don't crash on Windows cp1252
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim as ssim_fn
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ─── LOAD ENV ─────────────────────────────────────────────────
load_dotenv()

# Ensure project root is in sys.path so models/utils/export can be imported
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
NANODET_ROOT   = os.environ.get('NANODET_ROOT',   os.path.join(os.path.dirname(__file__), 'third_party', 'nanodet'))
RTDETR_WEIGHTS = os.environ.get('RTDETR_WEIGHTS', os.path.join('pretrained_models', 'rtdetr-l.pt'))
DATASET_ROOT   = os.environ.get('DATASET_ROOT',   os.path.join(os.path.dirname(__file__), 'data'))

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--game',       required=True,  type=str,
                    help='Game name (any name is accepted)')
parser.add_argument('--data_path',  default=None,   type=str,
                    help='Direct path to dataset folder (overrides DATASET_ROOT)')
parser.add_argument('--model',      required=True,  type=str,
                    choices=['yolov5n', 'nanodet', 'rtdetr'])
parser.add_argument('--n_iter',     default=100,    type=int)
parser.add_argument('--lr',         default=0.0005, type=float)
parser.add_argument('--epsilon',    default=8,      type=int)
parser.add_argument('--gpu',        default='0',    type=str)
parser.add_argument('--batch_size', default=8,      type=int)
parser.add_argument('--ssim_w',     default=0.3,    type=float,
                    help='Weight of SSIM loss (default: 0.3)')
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game
MODEL  = args.model

DATA_PATH   = args.data_path if args.data_path else os.path.join(DATASET_ROOT, GAME)

MODEL_SIZE  = {'yolov5n': 416, 'nanodet': 320, 'rtdetr': 640}
MODEL_INPUT = MODEL_SIZE[MODEL]

NOISE_OUT_DIR   = os.path.join('universal_cloak', GAME, MODEL)
os.makedirs(NOISE_OUT_DIR, exist_ok=True)
NOISE_SAVE_PATH = os.path.join(NOISE_OUT_DIR, 'universal_noise.pt')

RESULT_DIR  = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
TIMING_XLSX = os.path.join(RESULT_DIR, 'cloak_timing.xlsx')
TIMING_PNG  = os.path.join(RESULT_DIR, 'cloak_timing.png')

print(f"\n{'='*55}")
print(f"  Game  : {GAME}  |  Model : {MODEL}")
print(f"  Data  : {DATA_PATH}")
print(f"  Iter  : {args.n_iter}  LR : {args.lr}  EPS : {args.epsilon}/255")
print(f"  SSIM weight : {args.ssim_w}")
print(f"{'='*55}\n")


# ─── LOAD DATASET ────────────────────────────────────────────
def load_images_original(folder):
    exts    = ('.jpg', '.jpeg', '.png', '.bmp')
    paths   = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    tensors, sizes = [], []
    for p in paths:
        img = Image.open(p).convert('RGB')
        sizes.append(img.size)
        t = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
        tensors.append(t)
    return tensors, sizes


def letterbox_tensor(tensor, new_size, color=114/255, stride=32):
    """
    Replicates Ultralytics' LetterBox preprocessing exactly: scales the image
    to fit within (new_size, new_size) while preserving aspect ratio, then
    pads the remaining space with grey (114/255) split evenly on both sides.
    This must match what model.predict() does internally at eval time, or
    a universal noise trained on plain squish-resized frames will not align
    with the letterboxed frames used at inference (this was confirmed to be
    the cause of RT-DETR's near-zero DSR despite loss dropping in training).

    tensor: (3, H, W) float tensor in [0, 1]
    Returns: (padded (3,new_size,new_size), scale r, (pad_left, pad_top))
    """
    c, h, w = tensor.shape
    r = min(new_size / h, new_size / w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))
    resized = F.interpolate(tensor.unsqueeze(0), size=(new_unpad_h, new_unpad_w),
                            mode='bilinear', align_corners=False)[0]
    dw, dh = new_size - new_unpad_w, new_size - new_unpad_h
    dw, dh = dw / 2, dh / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    padded = F.pad(resized, (left, right, top, bottom), value=color)
    return padded, r, (left, top)


def to_model_input(tensor, model_input_size):
    if MODEL == 'rtdetr':
        padded, _, _ = letterbox_tensor(tensor, model_input_size)
        return padded.unsqueeze(0)
    img = tensor.unsqueeze(0)
    return F.interpolate(img, size=(model_input_size, model_input_size),
                         mode='bilinear', align_corners=False)


def noise_to_original(noise_model, orig_h, orig_w):
    n = noise_model.unsqueeze(0)
    return F.interpolate(n, size=(orig_h, orig_w),
                         mode='bilinear', align_corners=False).squeeze(0)


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_yolov5n():
    sys.path.insert(0, os.path.abspath('.'))
    print("  [YOLOv5n] Loading weights...", flush=True)
    from models.common import DetectMultiBackend
    model = DetectMultiBackend(
        os.path.join('pretrained_models', 'yolov5n.pt'),
        device=torch.device(DEVICE), fuse=True)
    model.eval()
    print("  [YOLOv5n] Loaded.", flush=True)
    return model


def load_nanodet():
    sys.path.insert(0, NANODET_ROOT)
    from nanodet.util import cfg, load_config, Logger as NL
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight
    load_config(cfg, os.path.join(NANODET_ROOT, 'config', 'nanodet-plus-m_320.yml'))
    model = build_model(cfg.model)
    ckpt  = torch.load(os.path.join(NANODET_ROOT, 'nanodet', 'nanodet-plus-m_320.pth'),
                       map_location='cpu')
    load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
    model = model.to(DEVICE).eval()
    print("  [NanoDet-Plus] Loaded.")
    return model, cfg


def load_rtdetr():
    from ultralytics import YOLO
    yolo  = YOLO(RTDETR_WEIGHTS)
    model = yolo.model.to(DEVICE).eval()
    # IMPORTANT: model.predict() always runs the model in eval() mode and reads
    # out[0] (shape (1, num_queries<=300, 6) = [x,y,w,h, conf, class_id]) — this
    # is confirmed directly from Ultralytics' RTDETRPredictor.postprocess():
    #   preds = preds[0]; bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
    # An earlier attempt switched to model.train() to target a deep-supervision
    # tensor (out[1]) that doesn't exist under eval() and isn't what .predict()
    # actually reads — that was a dead end. Training must match eval() exactly.
    print("  [RT-DETR] Loaded (eval mode, matching model.predict() exactly).")
    return model


# ─── LOSS FUNCTIONS ──────────────────────────────────────────
_bce = nn.BCELoss()


def loss_yolov5(model, adv_model):
    preds = model(adv_model)
    raw   = preds[0] if isinstance(preds, (list, tuple)) else preds
    conf  = torch.sigmoid(raw[..., 4:5])
    return _bce(conf, torch.zeros_like(conf))


def loss_nanodet(model, adv_model):
    mean = torch.tensor([103.53, 116.28, 123.675], device=DEVICE).view(1, 3, 1, 1) / 255.0
    std  = torch.tensor([57.375, 57.12,  58.395],  device=DEVICE).view(1, 3, 1, 1) / 255.0
    img_norm = (adv_model - mean) / std
    feats    = model.backbone(img_norm)
    feats    = model.fpn(feats)
    preds    = model.head(feats)
    total    = torch.tensor(0.0, device=DEVICE)
    for p in preds:
        if isinstance(p, (list, tuple)):
            p = p[0]
        prob  = torch.sigmoid(p[..., 0:1])
        total = total + _bce(prob, torch.zeros_like(prob))
    return total


def loss_rtdetr(model, adv_model):
    out = model(adv_model)
    if isinstance(out, dict) and 'pred_logits' in out:
        logits = out['pred_logits']
        person = torch.sigmoid(logits[:, :, 0:1])
        return _bce(person, torch.zeros_like(person))

    # model.eval() forward (see load_rtdetr) returns out[0] with shape
    # (batch, k<=300, 6) = [x, y, w, h, conf, class_id] — conf is already the
    # sigmoid'd, top-1-across-classes score, post top-k selection. This is
    # confirmed (via Ultralytics source, models/rtdetr/predict.py) to be the
    # EXACT tensor RTDETRPredictor.postprocess() reads as `preds[0]` to build
    # the confidence values that model.predict() reports — so gradients here
    # map directly onto real inference-time behavior, unlike deep-supervision
    # or auxiliary decoder tensors from other forward paths.
    #
    # Only a few of the k rows actually correspond to "person" (class 0); the
    # rest are top-scoring boxes of other classes. Averaging over all of them
    # dilutes the signal, so instead we take the single highest-confidence
    # person row per image (the one actually driving a real detection) and
    # push just that one toward 0.
    preds = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(preds, torch.Tensor) and preds.dim() == 3 and preds.shape[-1] == 6:
        conf        = preds[..., 4]                          # (batch, k) already sigmoid'd
        class_id    = preds[..., 5]                           # (batch, k)
        person_mask = (class_id < 0.5).float()                # rows whose top class is person
        person_conf = conf * person_mask                      # non-person rows zeroed out
        top_person_conf, _ = person_conf.max(dim=-1)           # (batch,) worst-case person conf per image
        top_person_conf    = top_person_conf.clamp(1e-6, 1 - 1e-6)
        return _bce(top_person_conf, torch.zeros_like(top_person_conf))

    # Fallback: deep-supervision-style stacked tensor (only exists in train() mode)
    if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], torch.Tensor) \
            and out[1].dim() == 4:
        class_logits = out[1]                                # (num_layers, batch, num_queries, num_classes)
        person_logit = class_logits[..., 0]                    # class 0 = person -> (num_layers, batch, num_queries)
        person_prob  = torch.sigmoid(person_logit)
        top_prob, _  = person_prob.max(dim=-1)                 # (num_layers, batch)
        top_prob     = top_prob.clamp(1e-6, 1 - 1e-6)
        return _bce(top_prob, torch.zeros_like(top_prob))

    # Fallback: eval-mode-style nested tuple (out[1][3] = last-layer-only logits)
    if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], (list, tuple)) \
            and len(out[1]) > 3 and isinstance(out[1][3], torch.Tensor):
        class_logits = out[1][3]
        person_logit = class_logits[..., 0]
        person_prob  = torch.sigmoid(person_logit)
        top_prob, _  = person_prob.max(dim=-1)
        top_prob     = top_prob.clamp(1e-6, 1 - 1e-6)
        return _bce(top_prob, torch.zeros_like(top_prob))

    # Last-resort fallback (older/newer ultralytics versions, different signature)
    raw  = out[0] if isinstance(out, (list, tuple)) else out
    conf     = raw[..., 4:5].clamp(1e-6, 1 - 1e-6)
    class_id = raw[..., 5:6]
    person_mask = (class_id < 0.5).float()
    weight      = person_mask * 4.0 + 1.0
    loss = -(weight * torch.log(1.0 - conf)).sum() / weight.sum()
    return loss


# ─── TRAINING ────────────────────────────────────────────────
def train_universal_cloak(img_tensors, img_sizes, model, model_type):
    S         = MODEL_INPUT
    noise     = torch.empty(3, S, S, device=DEVICE).uniform_(-EPS, EPS)
    noise.requires_grad_(True)
    optimizer = torch.optim.Adam([noise], lr=args.lr)

    N          = len(img_tensors)
    BS         = args.batch_size
    best_loss  = float('inf')
    best_noise = noise.detach().clone()
    losses     = []
    t0         = time.time()

    print(f"  Training {args.n_iter} iter | batch={BS} | {N} images | input={S}x{S}")

    for it in range(1, args.n_iter + 1):
        idx        = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, BS):
            batch_idx = idx[start:start + BS].tolist()
            batch = torch.stack([
                to_model_input(img_tensors[i], S)[0] for i in batch_idx
            ]).to(DEVICE)

            n_c = torch.clamp(noise, -EPS, EPS)
            adv = torch.clamp(batch + n_c.unsqueeze(0), 0.0, 1.0)

            if model_type == 'yolov5n':
                det_loss = loss_yolov5(model, adv)
            elif model_type == 'nanodet':
                det_loss = loss_nanodet(model, adv)
            elif model_type == 'rtdetr':
                det_loss = loss_rtdetr(model, adv)

            ssim_val  = ssim_fn((batch * 255).detach().cpu(),
                                (adv * 255).detach().cpu(),
                                data_range=255, size_average=True)
            ssim_loss = 1.0 - ssim_val.to(DEVICE)
            loss      = det_loss + args.ssim_w * ssim_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                noise.clamp_(-EPS, EPS)

            epoch_loss += det_loss.item()
            n_batches  += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        if avg < best_loss:
            best_loss  = avg
            best_noise = noise.detach().clone()

        # Progress bar + percentage (printed every iteration)
        pct      = it / args.n_iter
        bar_len  = 30
        filled   = int(bar_len * pct)
        bar      = '#' * filled + '-' * (bar_len - filled)
        elapsed  = time.time() - t0
        eta      = (elapsed / it) * (args.n_iter - it) if it > 0 else 0
        print(f"\r    [{bar}] {pct*100:5.1f}%  "
              f"Epoch {it}/{args.n_iter}  Loss={avg:.4f}  Best Loss={best_loss:.4f}  "
              f"Total Time={elapsed:.0f}s  Time Remaining={eta:.0f}s",
              end='', flush=True)

    print()  # newline after progress bar finishes

    total_time = time.time() - t0
    print(f"\n  Done — Total Time={total_time:.1f}s ({total_time / 60:.1f} min)  Best Loss={best_loss:.6f}")
    return best_noise.cpu(), total_time, losses


# ─── TIMING SAVE ─────────────────────────────────────────────
def save_timing(game, model_name, total_time_sec, n_images, n_iter):
    row = {'Game': game.upper(), 'Model': model_name, 'Images': n_images,
           'Iterations': n_iter, 'TotalTime_s': round(total_time_sec, 2),
           'TimePerIter_s': round(total_time_sec / n_iter, 4)}
    if os.path.exists(TIMING_XLSX):
        df = pd.read_excel(TIMING_XLSX)
        df = df[~((df['Game'] == row['Game']) & (df['Model'] == row['Model']))]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(TIMING_XLSX, index=False)
    print(f"  Timing → {TIMING_XLSX}")

    if len(df) >= 1:
        labels = [f"{r['Game']}\n{r['Model']}" for _, r in df.iterrows()]
        times  = df['TotalTime_s'].tolist()
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.0), 5))
        bars = ax.bar(labels, times, color='steelblue', width=0.55)
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(times) * 0.01,
                    f'{val:.0f}s', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Total Training Time (s)')
        ax.set_title('Universal Cloak — Training Time per Game x Model')
        ax.set_ylim(0, max(times) * 1.18)
        plt.tight_layout()
        plt.savefig(TIMING_PNG, dpi=150)
        plt.close()
        print(f"  Chart  → {TIMING_PNG}")


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("\n[1] Loading images (original size) ...")
    img_tensors, img_sizes = load_images_original(DATA_PATH)
    print(f"  Sample size: {img_sizes[0]}")

    print(f"\n[2] Loading model: {MODEL} ...")
    nano_cfg = None
    if MODEL == 'yolov5n':
        model = load_yolov5n()
    elif MODEL == 'nanodet':
        model, nano_cfg = load_nanodet()
    elif MODEL == 'rtdetr':
        model = load_rtdetr()

    print("\n[3] Training ...")
    best_noise, total_time, loss_curve = train_universal_cloak(
        img_tensors, img_sizes, model, MODEL)

    torch.save(best_noise, NOISE_SAVE_PATH)
    _noise_hash = hash(best_noise.numpy().tobytes()) & 0xFFFFFFFF
    print(f"\n[4] Noise saved → {NOISE_SAVE_PATH}")
    print(f"    Shape : {best_noise.shape}  (model input size {MODEL_INPUT}x{MODEL_INPUT})")
    print(f"    Range : [{best_noise.min():.4f}, {best_noise.max():.4f}]")
    print(f"    Fingerprint : mean_abs={best_noise.abs().mean().item():.6f}  "
          f"std={best_noise.std().item():.6f}  hash={_noise_hash:08x}")

    loss_png = os.path.join(NOISE_OUT_DIR, 'loss_curve.png')
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve, color='crimson')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.title(f'Loss Curve — {GAME.upper()} / {MODEL}')
    plt.tight_layout()
    plt.savefig(loss_png, dpi=150)
    plt.close()
    print(f"    Loss curve → {loss_png}")

    print("\n[5] Saving timing ...")
    save_timing(GAME, MODEL, total_time, len(img_tensors), args.n_iter)

    print(f"\n{'='*55}")
    print(f"  COMPLETE : {GAME.upper()} / {MODEL}")
    print(f"  Total Time : {total_time:.1f}s  ({total_time / 60:.1f} min)")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()