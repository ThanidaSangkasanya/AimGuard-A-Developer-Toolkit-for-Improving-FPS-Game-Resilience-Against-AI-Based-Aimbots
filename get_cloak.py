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
                    choices=['yolov5n', 'nanodet', 'rtdetr'])
parser.add_argument('--epsilon',      default=8,     type=int)
parser.add_argument('--conf',         default=0.4,   type=float,
                    help='Confidence threshold for DSR (default: 0.4)')


parser.add_argument('--gpu',          default='0',   type=str)
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'
EPS    = args.epsilon / 255.0
GAME   = args.game
MODEL  = args.model

MODEL_SIZE  = {'yolov5n': 416, 'nanodet': 320, 'rtdetr': 640}
MODEL_INPUT = MODEL_SIZE[MODEL]

DATA_PATH  = args.data_path if args.data_path else os.path.join(DATASET_ROOT, GAME)
NOISE_PATH = os.path.join('universal_cloak', GAME, MODEL, 'universal_noise.pt')

# DSR and FPS only — no Recall/GT required

ATTACK_DIR       = os.path.join('result', 'evaluation', GAME, MODEL, 'attack')
ATTACK_CLEAN_DIR = os.path.join('result', 'evaluation', GAME, MODEL, 'attack_clean')
LOG_DIR          = os.path.join('result', 'log', GAME)
os.makedirs(ATTACK_DIR,       exist_ok=True)
os.makedirs(ATTACK_CLEAN_DIR, exist_ok=True)
os.makedirs(LOG_DIR,          exist_ok=True)

EVAL_XLSX = os.path.join('result', 'evaluation_summary.xlsx')
EVAL_PNG  = os.path.join('result', 'evaluation_summary.png')

print(f"\n{'='*55}")
print(f"  Game  : {GAME}  |  Model : {MODEL}")
print(f"  Data  : {DATA_PATH}")
print(f"  Noise : {NOISE_PATH}")
print(f"{'='*55}\n")

# Load noise
if not os.path.exists(NOISE_PATH):
    raise FileNotFoundError(f"Noise not found: {NOISE_PATH}\nRun train_cloak.py first.")
noise_model_size = torch.load(NOISE_PATH).to(DEVICE)
print(f"  Noise loaded: {noise_model_size.shape}")





# ─── LOAD DATASET ────────────────────────────────────────────
def load_image_paths(folder):
    exts  = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(exts)])
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")
    print(f"  Found {len(paths)} images")
    return paths


# ─── APPLY NOISE ─────────────────────────────────────────────
def apply_noise_original(img_pil):
    orig_w, orig_h = img_pil.size
    img_t = torch.from_numpy(
        np.array(img_pil).transpose(2, 0, 1)
    ).float().unsqueeze(0).to(DEVICE) / 255.0

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
def detect_yolov5(model, adv_pil, conf_thr=None):
    if conf_thr is None:
        conf_thr = args.conf
    adv_np     = np.array(adv_pil)
    orig_conf  = model.conf
    model.conf = conf_thr
    with torch.no_grad():
        results = model(adv_np, size=MODEL_INPUT)
    model.conf = orig_conf
    boxes = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0:
            boxes.append([int(xyxy[0]), int(xyxy[1]),
                          int(xyxy[2]), int(xyxy[3]), float(conf)])
    return boxes


def detect_nanodet(model, nano_cfg, pipeline, adv_pil, conf_thr=None):
    if conf_thr is None:
        conf_thr = args.conf
    import cv2
    from nanodet.data.collate import naive_collate
    from nanodet.data.batch_process import stack_batch_img
    adv_np  = np.array(adv_pil)
    img_bgr = cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR)
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


def detect_rtdetr(model, adv_pil, conf_thr=None):
    if conf_thr is None:
        conf_thr = args.conf
    adv_np  = np.array(adv_pil)
    results = model.predict(source=adv_np, conf=conf_thr,
                            classes=[0], verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append([x1, y1, x2, y2, float(box.conf[0])])
    return boxes


# ─── DRAW BOXES ──────────────────────────────────────────────
def draw_boxes(pil_img, pred_boxes):
    """Draw predicted bounding boxes (red) on cloaked frame."""
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2, conf) in pred_boxes:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1 - 15)), f'{conf:.2f}', fill='red')
    return img


# ─── SUMMARY SAVE ────────────────────────────────────────────
def save_summary(game, model_name, dsr, avg_fps, n_images, total_time_s):
    import csv as _csv
    row = {
        'Game':        game.upper(),
        'Model':       model_name,
        'Images':      n_images,
        'DSR':         round(dsr,     4),
        'AvgFPS':      round(avg_fps, 2),
        'TotalTime_s': round(total_time_s, 2),
    }

    summary_csv = EVAL_XLSX.replace('.xlsx', '.csv')
    fieldnames  = ['Game', 'Model', 'Images', 'DSR', 'AvgFPS', 'TotalTime_s']

    # Read existing rows (csv), drop matching game+model
    rows = []
    if os.path.exists(summary_csv):
        try:
            with open(summary_csv, 'r', newline='') as f:
                for r in _csv.DictReader(f):
                    if not (r.get('Game') == row['Game'] and r.get('Model') == row['Model']):
                        rows.append(r)
        except Exception:
            rows = []
    rows.append(row)

    # Write back
    try:
        with open(summary_csv, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Summary → {summary_csv}", flush=True)
    except Exception as e:
        print(f"  [warn] could not save summary: {e}", flush=True)

    # Chart
    try:
        labels = [f"{r['Game']}\n{r['Model']}" for r in rows]
        dsrs   = [float(r['DSR']) * 100 for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 5))
        bars = ax.bar(labels, dsrs, color='steelblue', width=0.5)
        for bar, val in zip(bars, dsrs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('DSR (%)')
        ax.set_ylim(0, 115)
        ax.set_title('AimGuard — Defense Success Rate per Game x Model')
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
    nano_cfg = nano_pipeline = None
    if MODEL == 'yolov5n':
        detect_model = load_yolov5n()
    elif MODEL == 'nanodet':
        detect_model, nano_cfg, nano_pipeline = load_nanodet()
    elif MODEL == 'rtdetr':
        detect_model = load_rtdetr()

    print(f"\n[3] Evaluating {len(img_paths)} images ...")
    succ_num   = 0
    total_time = 0.0
    log_rows   = []

    for i, img_path in enumerate(img_paths):
        img_pil        = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size
        fname          = os.path.splitext(os.path.basename(img_path))[0]

        t_start = time.time()
        adv_pil, adv_t, img_t = apply_noise_original(img_pil)

        if MODEL == 'yolov5n':
            pred_boxes = detect_yolov5(detect_model, adv_pil, args.conf)
        elif MODEL == 'nanodet':
            pred_boxes = detect_nanodet(detect_model, nano_cfg, nano_pipeline, adv_pil, args.conf)
        elif MODEL == 'rtdetr':
            pred_boxes = detect_rtdetr(detect_model, adv_pil, args.conf)

        t_elapsed   = time.time() - t_start
        total_time += t_elapsed

        target_succ = 1 if len(pred_boxes) == 0 else 0
        succ_num   += target_succ

        fps     = (i + 1) / total_time
        dsr_now = succ_num / (i + 1)

        print(f"  [{i+1:3d}/{len(img_paths)}] "
              f"Pred:{len(pred_boxes)} "
              f"DSR:{dsr_now:.3f} FPS:{fps:.1f}", flush=True)

        # Save clean (no boxes) and with pred boxes (red) for visualization
        try:
            adv_pil.convert('RGB').save(os.path.join(ATTACK_CLEAN_DIR, f'{fname}_attack.jpg'))
            adv_vis = draw_boxes(adv_pil, pred_boxes)
            adv_vis.convert('RGB').save(os.path.join(ATTACK_DIR, f'{fname}_attack.jpg'))
        except Exception as e:
            print(f"    [warn] could not save visualization for {fname}: {e}", flush=True)

        log_rows.append({
            'Index':      i,
            'File':       os.path.basename(img_path),
            'Pred_boxes': len(pred_boxes),
            'TSucc':      target_succ,
            'DSR':        round(dsr_now,   4),
            'Time_s':     round(t_elapsed, 4),
            'FPS':        round(fps,       2),
        })

        # Free per-iteration GPU tensors
        del adv_t, img_t

    print("  [debug] loop finished, starting cleanup...", flush=True)

    # Free GPU memory before file I/O to prevent crash on some systems
    try:
        del detect_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("  [debug] gpu cleanup done", flush=True)

    print("  [debug] saving log with csv module...", flush=True)
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
    final_fps = len(img_paths) / total_time
    total_min = total_time / 60

    print(f"\n{'='*55}", flush=True)
    print(f"  RESULTS : {GAME.upper()} / {MODEL}", flush=True)
    print(f"  Total runtime : {total_time:.1f}s ({total_min:.1f} min)", flush=True)
    print(f"  DSR     : {final_dsr:.4f} ({final_dsr*100:.1f}%)", flush=True)
    print(f"  AvgFPS  : {final_fps:.2f}", flush=True)
    print(f"{'='*55}\n", flush=True)

    save_summary(GAME, MODEL, final_dsr, final_fps, len(img_paths), total_time)
    print("  Summary saved. Done.", flush=True)


if __name__ == '__main__':
    main()