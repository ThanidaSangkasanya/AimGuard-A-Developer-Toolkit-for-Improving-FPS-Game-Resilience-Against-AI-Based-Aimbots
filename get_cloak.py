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
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ─── LOAD ENV ─────────────────────────────────────────────────
load_dotenv()
NANODET_ROOT   = os.environ.get('NANODET_ROOT',   os.path.join(os.path.dirname(__file__), 'third_party', 'nanodet'))
RTDETR_WEIGHTS = os.environ.get('RTDETR_WEIGHTS', os.path.join('pretrained_models', 'rtdetr-l.pt'))
DATASET_ROOT   = os.environ.get('DATASET_ROOT',   os.path.join(os.path.dirname(__file__), 'data'))


# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--game',         required=True, type=str,
                    help='Game name (any name is accepted)')
parser.add_argument('--data_path',    default=None,  type=str,
                    help='Direct path to dataset folder (overrides DATASET_ROOT)')
parser.add_argument('--model',        required=True, type=str,
                    choices=['yolov5n', 'nanodet', 'rtdetr', 'custom'])
parser.add_argument('--custom_model_path', default=None, type=str,
                    help='Required when --model custom is used. Must be a YOLO-compatible '
                         '(Ultralytics YOLOv5/YOLOv8 style) .pt weight file.')
parser.add_argument('--epsilon',      default=8,     type=int)
parser.add_argument('--conf',         default=0.4,   type=float,
                    help='Confidence threshold for DSR (default: 0.4)')
parser.add_argument('--label_path',   default=None,  type=str,
                    help='Optional path to ground-truth labels (YOLO .txt format, '
                         'one file per image, same basename as the image). '
                         'If provided, Recall Before/After/Drop are computed in addition to DSR.')
parser.add_argument('--iou_thr',      default=0.5,   type=float,
                    help='IoU threshold used to match predicted boxes to ground-truth boxes '
                         '(only used when --label_path is provided)')
parser.add_argument('--save_gallery', default=True,  type=lambda v: str(v).lower() != 'false',
                    help='Whether to save before/after clean & bbox images for the comparison '
                         'gallery (default: True)')


parser.add_argument('--gpu',          default='0',   type=str)
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game
MODEL  = args.model

MODEL_SIZE  = {'yolov5n': 416, 'nanodet': 320, 'rtdetr': 640, 'custom': 416}
MODEL_INPUT = MODEL_SIZE[MODEL]

DATA_PATH  = args.data_path if args.data_path else os.path.join(DATASET_ROOT, GAME)
NOISE_PATH = os.path.join('universal_cloak', GAME, MODEL, 'universal_noise.pt')
LABEL_PATH = args.label_path
USE_GT     = LABEL_PATH is not None and os.path.isdir(LABEL_PATH)

# ─── OUTPUT DIRECTORIES ───────────────────────────────────────
EVAL_ROOT       = os.path.join('result', 'evaluation', GAME, MODEL)
BEFORE_CLEAN_DIR = os.path.join(EVAL_ROOT, 'before_clean')   # raw frame, no boxes
BEFORE_BBOX_DIR  = os.path.join(EVAL_ROOT, 'before_bbox')    # raw frame + red pred + green GT
AFTER_CLEAN_DIR  = os.path.join(EVAL_ROOT, 'after_clean')    # cloaked frame, no boxes
AFTER_BBOX_DIR   = os.path.join(EVAL_ROOT, 'after_bbox')     # cloaked frame + red pred + green GT
LOG_DIR          = os.path.join('result', 'log', GAME)
for d in (BEFORE_CLEAN_DIR, BEFORE_BBOX_DIR, AFTER_CLEAN_DIR, AFTER_BBOX_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

EVAL_XLSX = os.path.join('result', 'evaluation_summary.xlsx')
EVAL_PNG  = os.path.join('result', 'evaluation_summary.png')

print(f"\n{'='*55}")
print(f"  Game  : {GAME}  |  Model : {MODEL}")
print(f"  Data  : {DATA_PATH}")
print(f"  Noise : {NOISE_PATH}")
if USE_GT:
    print(f"  Labels: {LABEL_PATH}  (IoU thr={args.iou_thr})")
else:
    print(f"  Labels: none — Recall Before/After/Drop will be skipped")
print(f"{'='*55}\n")

# Load noise
if not os.path.exists(NOISE_PATH):
    raise FileNotFoundError(f"Noise not found: {NOISE_PATH}\nRun train_cloak.py first.")
noise_model_size = torch.load(NOISE_PATH).to(DEVICE)
print(f"  Noise loaded: {noise_model_size.shape}")
_noise_hash = hash(noise_model_size.detach().cpu().numpy().tobytes()) & 0xFFFFFFFF
print(f"  Fingerprint : mean_abs={noise_model_size.abs().mean().item():.6f}  "
     f"std={noise_model_size.std().item():.6f}  hash={_noise_hash:08x}  "
     f"(mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(NOISE_PATH)))})")


# ─── LOAD DATASET ────────────────────────────────────────────
def load_image_paths(folder):
    exts  = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    return paths


# ─── GROUND-TRUTH LABEL LOADING (YOLO format) ────────────────
def load_gt_boxes(img_path, orig_w, orig_h):
    """
    Loads YOLO-format ground truth boxes for a given image, if a matching
    label file exists in LABEL_PATH. Each line: class x_center y_center w h
    (all normalized 0-1). Returns a list of [x1, y1, x2, y2] pixel boxes,
    person class (0) only. Returns None if no label file is found for
    this image (as opposed to an empty list, which means "no objects").
    """
    if not USE_GT:
        return None
    fname     = os.path.splitext(os.path.basename(img_path))[0]
    label_txt = os.path.join(LABEL_PATH, f'{fname}.txt')
    if not os.path.exists(label_txt):
        return None

    boxes = []
    try:
        with open(label_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = parts[:5]
                cls = int(float(cls))
                if cls != 0:
                    continue
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                x1 = (xc - w / 2) * orig_w
                y1 = (yc - h / 2) * orig_h
                x2 = (xc + w / 2) * orig_w
                y2 = (yc + h / 2) * orig_h
                boxes.append([x1, y1, x2, y2])
    except Exception as e:
        print(f"    [warn] could not parse label {label_txt}: {e}", flush=True)
        return None
    return boxes


def box_iou(box_a, box_b):
    """IoU between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w  = max(0.0, inter_x2 - inter_x1)
    inter_h  = max(0.0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h
    area_a   = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b   = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union    = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_thr):
    """
    Greedy IoU matching between predicted boxes (list of [x1,y1,x2,y2,conf])
    and ground-truth boxes (list of [x1,y1,x2,y2]).
    Returns (tp, fp, fn) counts for this image.
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    preds_sorted = sorted(pred_boxes, key=lambda b: b[4], reverse=True)
    for p in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = box_iou(p[:4], gt)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_thr and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ─── APPLY NOISE ─────────────────────────────────────────────
def unletterbox_noise(noise_model_frame, orig_h, orig_w, stride=32):
    """
    Inverse of the letterbox transform used during training for RT-DETR:
    crops out the grey padding region the noise was trained against, then
    rescales the remaining "active" region back up to the original image's
    resolution. This keeps the noise spatially aligned with what
    model.predict() will see after it letterboxes the noised original image
    again internally at inference time.
    """
    new_size = noise_model_frame.shape[-1]  # square model input, e.g. 640
    r = min(new_size / orig_h, new_size / orig_w)
    new_unpad_w, new_unpad_h = int(round(orig_w * r)), int(round(orig_h * r))
    dw, dh = new_size - new_unpad_w, new_size - new_unpad_h
    dw, dh = dw / 2, dh / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))

    cropped = noise_model_frame[:, top:top + new_unpad_h, left:left + new_unpad_w]
    noise_orig = F.interpolate(
        cropped.unsqueeze(0), size=(orig_h, orig_w),
        mode='bilinear', align_corners=False
    ).squeeze(0)
    return noise_orig


def apply_noise_original(img_pil):
    orig_w, orig_h = img_pil.size
    img_t = torch.from_numpy(
        np.array(img_pil).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(DEVICE) / 255.0

    if MODEL == 'rtdetr':
        # noise_model_size lives in a letterboxed 640x640 frame (matching
        # how it was trained) — map it back to original-image coordinates
        # correctly instead of a naive squish-resize.
        noise_orig = unletterbox_noise(noise_model_size, orig_h, orig_w)
    else:
        noise_orig = F.interpolate(
            noise_model_size.unsqueeze(0),
            size=(orig_h, orig_w),
            mode='bilinear', align_corners=False
        ).squeeze(0)

    noise_clamped = torch.clamp(noise_orig, -EPS, EPS)
    adv_t  = torch.clamp(img_t + noise_clamped.unsqueeze(0), 0.0, 1.0)
    adv_np = (adv_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_np)
    return adv_pil, adv_t, img_t


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_yolov5n():
    sys.path.insert(0, os.path.abspath('.'))
    print("  [YOLOv5n] Loading weights...", flush=True)
    from models.common import DetectMultiBackend, AutoShape
    model = DetectMultiBackend(
        os.path.join('pretrained_models', 'yolov5n.pt'),
        device=torch.device(DEVICE), fuse=True)
    model = AutoShape(model)
    model.amp     = False
    model.conf    = args.conf
    model.classes = [0]
    print("  [YOLOv5n] Loaded.", flush=True)
    return model


def load_custom():
    if not args.custom_model_path or not os.path.exists(args.custom_model_path):
        raise FileNotFoundError(
            f"--custom_model_path not found: {args.custom_model_path}")
    sys.path.insert(0, os.path.abspath('.'))
    print(f"  [Custom] Loading weights from {args.custom_model_path} ...", flush=True)
    from models.common import DetectMultiBackend, AutoShape
    model = DetectMultiBackend(
        args.custom_model_path, device=torch.device(DEVICE), fuse=True)
    model = AutoShape(model)
    model.amp     = False
    model.conf    = args.conf
    model.classes = [0]
    print("  [Custom] Loaded.", flush=True)
    return model


def load_nanodet():
    sys.path.insert(0, NANODET_ROOT)
    from nanodet.util import cfg, load_config, Logger as NL
    from nanodet.model.arch import build_model
    from nanodet.util import load_model_weight
    from nanodet.data.transform import Pipeline
    load_config(cfg, os.path.join(NANODET_ROOT, 'config', 'nanodet-plus-m_320.yml'))
    model = build_model(cfg.model)
    ckpt  = torch.load(os.path.join(NANODET_ROOT, 'nanodet', 'nanodet-plus-m_320.pth'),
                       map_location='cpu')
    load_model_weight(model, ckpt, NL(0, use_tensorboard=False))
    model    = model.to(DEVICE).eval()
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    print("  [NanoDet-Plus] Loaded.")
    return model, cfg, pipeline


def load_rtdetr():
    from ultralytics import YOLO
    model = YOLO(RTDETR_WEIGHTS)
    print("  [RT-DETR] Loaded.")
    return model


# ─── DETECTION ───────────────────────────────────────────────
def detect_yolov5(model, img_pil, conf_thr=None):
    if conf_thr is None:
        conf_thr = args.conf
    img_np     = np.array(img_pil)
    orig_conf  = model.conf
    model.conf = conf_thr
    with torch.no_grad():
        results = model(img_np, size=MODEL_INPUT)
    model.conf = orig_conf
    boxes = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0:
            boxes.append([int(xyxy[0]), int(xyxy[1]),
                          int(xyxy[2]), int(xyxy[3]), float(conf)])
    return boxes


def detect_nanodet(model, nano_cfg, pipeline, img_pil, conf_thr=None):
    if conf_thr is None:
        conf_thr = args.conf
    import cv2
    from nanodet.data.collate import naive_collate
    from nanodet.data.batch_process import stack_batch_img
    img_np  = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]
    meta = dict(img_info={'id': 0, 'file_name': None, 'height': h, 'width': w},
                raw_img=img_bgr, img=img_bgr)
    meta = pipeline(None, meta, nano_cfg.data.val.input_size)
    meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).to(DEVICE)
    meta = naive_collate([meta])
    meta['img'] = stack_batch_img(meta['img'], divisible=32)
    with torch.no_grad():
        results = model.inference(meta)
    boxes = []
    for d in results[0].get(0, []):
        if d[-1] >= conf_thr:
            boxes.append([int(d[0]), int(d[1]), int(d[2]), int(d[3]), float(d[-1])])
    return boxes


def detect_rtdetr(model, img_pil, conf_thr=None):
    if conf_thr is None:
        conf_thr = args.conf
    img_np  = np.array(img_pil)
    results = model.predict(source=img_np, conf=conf_thr,
                            classes=[0], verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append([x1, y1, x2, y2, float(box.conf[0])])
    return boxes


def run_detection(model_bundle, img_pil, conf_thr=None):
    """Dispatch to the correct detect_* function based on MODEL."""
    if MODEL in ('yolov5n', 'custom'):
        return detect_yolov5(model_bundle, img_pil, conf_thr)
    elif MODEL == 'nanodet':
        model, nano_cfg, nano_pipeline = model_bundle
        return detect_nanodet(model, nano_cfg, nano_pipeline, img_pil, conf_thr)
    elif MODEL == 'rtdetr':
        return detect_rtdetr(model_bundle, img_pil, conf_thr)


# ─── DRAW BOXES ──────────────────────────────────────────────
def draw_boxes(pil_img, pred_boxes, gt_boxes=None):
    """Draw predicted bounding boxes (red) and, if available, ground-truth
    boxes (green) on a frame."""
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    if gt_boxes:
        for (x1, y1, x2, y2) in gt_boxes:
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=2)
    for (x1, y1, x2, y2, conf) in pred_boxes:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1 - 15)), f'{conf:.2f}', fill='red')
    return img


# ─── SUMMARY SAVE ────────────────────────────────────────────
def save_summary(game, model_name, dsr, n_images,
                  recall_before=None, recall_after=None, precision_after=None,
                  gt_images=0):
    import csv as _csv
    recall_drop = None
    if recall_before is not None and recall_after is not None:
        recall_drop = recall_before - recall_after

    row = {
        'Game':           game.upper(),
        'Model':          model_name,
        'Images':         n_images,
        'DSR':            round(dsr, 4),
        'Recall_Before':  round(recall_before, 4)   if recall_before   is not None else '',
        'Recall_After':   round(recall_after, 4)    if recall_after    is not None else '',
        'Recall_Drop':    round(recall_drop, 4)     if recall_drop     is not None else '',
        'Precision_After':round(precision_after, 4) if precision_after is not None else '',
        'GT_Images':      gt_images,
    }

    summary_csv = EVAL_XLSX.replace('.xlsx', '.csv')
    fieldnames  = ['Game', 'Model', 'Images', 'DSR', 'Recall_Before', 'Recall_After',
                   'Recall_Drop', 'Precision_After', 'GT_Images']

    rows = []
    if os.path.exists(summary_csv):
        try:
            with open(summary_csv, 'r', newline='') as f:
                for r in _csv.DictReader(f):
                    if not (r.get('Game') == row['Game'] and r.get('Model') == row['Model']):
                        for fn in fieldnames:
                            r.setdefault(fn, '')
                        rows.append(r)
        except Exception:
            rows = []
    rows.append(row)

    try:
        with open(summary_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Summary → {summary_csv}", flush=True)
    except PermissionError as e:
        # File is very likely open in Excel or another program and locked for
        # writing. Falling back silently would mean the dashboard keeps
        # showing STALE results from a previous run without anyone noticing —
        # so instead we write to a clearly-named fallback file AND print a
        # loud, impossible-to-miss warning.
        import time as _time
        fallback = summary_csv.replace('.csv', f'_UNSAVED_{int(_time.time())}.csv')
        try:
            with open(fallback, 'w', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception:
            fallback = None
        print(f"\n{'!'*70}", flush=True)
        print(f"  ⚠️  COULD NOT SAVE {summary_csv} — PERMISSION DENIED", flush=True)
        print(f"  ⚠️  The file is likely open in Excel or another program.", flush=True)
        print(f"  ⚠️  THE DASHBOARD IS STILL SHOWING OLD/STALE RESULTS FROM", flush=True)
        print(f"  ⚠️  A PREVIOUS RUN — THIS RUN'S NUMBERS WERE NOT SAVED THERE.", flush=True)
        if fallback:
            print(f"  ⚠️  This run's real results were saved to: {fallback}", flush=True)
        print(f"  ⚠️  Close the file in Excel, then re-run evaluation to fix this.", flush=True)
        print(f"{'!'*70}\n", flush=True)
    except Exception as e:
        print(f"  [warn] could not save summary: {e}", flush=True)

    # Chart: grouped bars — DSR, Recall Before, Recall After (each its own color)
    try:
        import numpy as _np

        labels = [f"{r['Game']}\n{r['Model']}" for r in rows]
        dsrs   = [float(r['DSR']) * 100 for r in rows]
        has_gt = [str(r.get('Recall_After', '')).strip() not in ('', None) for r in rows]

        x = _np.arange(len(rows))

        if any(has_gt):
            recalls_before = [float(r['Recall_Before']) * 100
                              if str(r.get('Recall_Before', '')).strip() not in ('', None) else 0
                              for r in rows]
            recalls_after  = [float(r['Recall_After']) * 100
                              if str(r.get('Recall_After', '')).strip() not in ('', None) else 0
                              for r in rows]

            width = 0.25
            fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.8), 5))
            bars_dsr    = ax.bar(x - width, dsrs,           width, color='steelblue', label='DSR (%)')
            bars_before = ax.bar(x,         recalls_before, width, color='crimson',   label='Recall Before (%)')
            bars_after  = ax.bar(x + width, recalls_after,  width, color='darkorange',label='Recall After (%)')

            for bars in (bars_dsr, bars_before, bars_after):
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1, f'{bar.get_height():.1f}%',
                            ha='center', va='bottom', fontsize=8)

            ax.legend(loc='upper right', fontsize=8)
        else:
            width = 0.5
            fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
            bars_dsr = ax.bar(x, dsrs, width, color='steelblue', label='DSR (%)')
            for bar in bars_dsr:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1, f'{bar.get_height():.1f}%',
                        ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Percent (%)')
        ax.set_ylim(0, 115)
        ax.set_title('AimGuard — DSR & Recall Before/After Cloak')
        plt.tight_layout()
        plt.savefig(EVAL_PNG, dpi=150)
        plt.close()
        print(f"  Chart  → {EVAL_PNG}", flush=True)
    except Exception as e:
        print(f"  [warn] could not save chart: {e}", flush=True)


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("\n[1] Loading image paths ...")
    img_paths = load_image_paths(DATA_PATH)

    print(f"\n[2] Loading model: {MODEL} ...")
    if MODEL == 'yolov5n':
        model_bundle = load_yolov5n()
    elif MODEL == 'custom':
        model_bundle = load_custom()
    elif MODEL == 'nanodet':
        model_bundle = load_nanodet()   # (model, cfg, pipeline)
    elif MODEL == 'rtdetr':
        model_bundle = load_rtdetr()

    print(f"\n[3] Evaluating {len(img_paths)} images (before vs after cloak) ...")
    succ_num = 0
    log_rows = []

    # Ground-truth accumulators (before = raw frame, after = cloaked frame)
    total_tp_before, total_fp_before, total_fn_before = 0, 0, 0
    total_tp_after,  total_fp_after,  total_fn_after  = 0, 0, 0
    gt_images_used = 0

    for i, img_path in enumerate(img_paths):
        img_pil        = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        fname          = os.path.splitext(os.path.basename(img_path))[0]

        # ── BEFORE: detection on the raw, unprotected frame ──
        pred_before = run_detection(model_bundle, img_pil, args.conf)

        # ── AFTER: apply Invisibility Cloak, then detect ─────
        adv_pil, adv_t, img_t = apply_noise_original(img_pil)
        pred_after = run_detection(model_bundle, adv_pil, args.conf)

        target_succ = 1 if len(pred_after) == 0 else 0
        succ_num   += target_succ
        dsr_now     = succ_num / (i + 1)

        # ── Ground-truth matching (optional) ──────────────
        gt_boxes = load_gt_boxes(img_path, orig_w, orig_h)
        img_tp_b = img_fp_b = img_fn_b = None
        img_tp_a = img_fp_a = img_fn_a = None
        if gt_boxes is not None:
            gt_images_used += 1
            img_tp_b, img_fp_b, img_fn_b = match_boxes(pred_before, gt_boxes, args.iou_thr)
            img_tp_a, img_fp_a, img_fn_a = match_boxes(pred_after,  gt_boxes, args.iou_thr)
            total_tp_before += img_tp_b; total_fp_before += img_fp_b; total_fn_before += img_fn_b
            total_tp_after  += img_tp_a; total_fp_after  += img_fp_a; total_fn_after  += img_fn_a

        gt_str = ""
        if gt_boxes is not None:
            rb = total_tp_before / (total_tp_before + total_fn_before) if (total_tp_before + total_fn_before) > 0 else 0.0
            ra = total_tp_after  / (total_tp_after  + total_fn_after)  if (total_tp_after  + total_fn_after)  > 0 else 0.0
            gt_str = f" RecallBefore:{rb:.3f} RecallAfter:{ra:.3f}"

        print(f"  [{i+1:3d}/{len(img_paths)}] "
              f"PredBefore:{len(pred_before)} PredAfter:{len(pred_after)} "
              f"DSR:{dsr_now:.3f}{gt_str}", flush=True)

        # ── Save before/after, clean/bbox gallery images ──
        if args.save_gallery:
            try:
                img_pil.convert('RGB').save(os.path.join(BEFORE_CLEAN_DIR, f'{fname}_before.jpg'))
                before_vis = draw_boxes(img_pil, pred_before, gt_boxes)
                before_vis.convert('RGB').save(os.path.join(BEFORE_BBOX_DIR, f'{fname}_before_bbox.jpg'))

                adv_pil.convert('RGB').save(os.path.join(AFTER_CLEAN_DIR, f'{fname}_after.jpg'))
                after_vis = draw_boxes(adv_pil, pred_after, gt_boxes)
                after_vis.convert('RGB').save(os.path.join(AFTER_BBOX_DIR, f'{fname}_after_bbox.jpg'))
            except Exception as e:
                print(f"    [warn] could not save gallery images for {fname}: {e}", flush=True)

        log_rows.append({
            'Index':        i,
            'File':         os.path.basename(img_path),
            'Pred_Before':  len(pred_before),
            'Pred_After':   len(pred_after),
            'GT_boxes':     len(gt_boxes) if gt_boxes is not None else '',
            'TP_Before':    img_tp_b if img_tp_b is not None else '',
            'FP_Before':    img_fp_b if img_fp_b is not None else '',
            'FN_Before':    img_fn_b if img_fn_b is not None else '',
            'TP_After':     img_tp_a if img_tp_a is not None else '',
            'FP_After':     img_fp_a if img_fp_a is not None else '',
            'FN_After':     img_fn_a if img_fn_a is not None else '',
            'TSucc':        target_succ,
            'DSR':          round(dsr_now, 4),
        })

        del adv_t, img_t

    print("  [debug] loop finished, starting cleanup...", flush=True)

    try:
        del model_bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("  [debug] gpu cleanup done", flush=True)

    log_path = os.path.join(LOG_DIR, f'{GAME}_{MODEL}_eval.csv')
    try:
        import csv as _csv
        if log_rows:
            with open(log_path, 'w', newline='') as f:
                writer = _csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
        print(f"\n  Log → {log_path}", flush=True)
    except Exception as e:
        print(f"  [warn] could not save log: {e}", flush=True)

    final_dsr = succ_num / len(img_paths)

    final_recall_before = total_tp_before / (total_tp_before + total_fn_before) \
        if (total_tp_before + total_fn_before) > 0 else None
    final_recall_after  = total_tp_after / (total_tp_after + total_fn_after) \
        if (total_tp_after + total_fn_after) > 0 else None
    final_precision_after = total_tp_after / (total_tp_after + total_fp_after) \
        if (total_tp_after + total_fp_after) > 0 else None
    final_recall_drop = (final_recall_before - final_recall_after) \
        if (final_recall_before is not None and final_recall_after is not None) else None

    print(f"\n{'='*55}", flush=True)
    print(f"  RESULTS : {GAME.upper()} / {MODEL}", flush=True)
    print(f"  DSR     : {final_dsr:.4f} ({final_dsr*100:.1f}%)", flush=True)
    if gt_images_used > 0:
        print(f"  GT images used : {gt_images_used}/{len(img_paths)}", flush=True)
        print(f"  Recall Before  : {final_recall_before:.4f} ({final_recall_before*100:.1f}%)", flush=True)
        print(f"  Recall After   : {final_recall_after:.4f} ({final_recall_after*100:.1f}%)", flush=True)
        print(f"  Recall Drop    : {final_recall_drop:.4f} ({final_recall_drop*100:.1f} pts)", flush=True)
        print(f"  Precision After: {final_precision_after:.4f} ({final_precision_after*100:.1f}%)", flush=True)
    elif USE_GT:
        print(f"  [warn] Ground-truth path given but no matching label files were found.", flush=True)
    print(f"{'='*55}\n", flush=True)

    save_summary(GAME, MODEL, final_dsr, len(img_paths),
                recall_before=final_recall_before, recall_after=final_recall_after,
                precision_after=final_precision_after, gt_images=gt_images_used)
    print("  Summary saved. Done.", flush=True)


if __name__ == '__main__':
    main()