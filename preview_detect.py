import os, sys, argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import torch
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()
NANODET_ROOT = os.environ.get('NANODET_ROOT', os.path.join(os.path.dirname(__file__), 'third_party', 'nanodet'))

# ─── ARGUMENTS ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run a detector on a RAW (uncloaked) frame to preview what a visual "
                "aimbot would see before Invisibility Cloak protection is applied."
)
parser.add_argument('--image',             required=True, type=str)
parser.add_argument('--model',             required=True, type=str,
                    choices=['yolov5n', 'nanodet', 'custom'])
parser.add_argument('--custom_model_path', default=None,  type=str,
                    help='Required when --model custom is used. Must be a YOLO-compatible '
                         '(Ultralytics YOLOv5/YOLOv8 style) .pt weight file.')
parser.add_argument('--conf',              default=0.4,   type=float)
parser.add_argument('--input_size',        default=None,  type=int,
                    help='Detection input resolution. Defaults per model: '
                         'yolov5n/custom=416, nanodet=320')
parser.add_argument('--out_dir',           default=os.path.join('result', 'preview'), type=str)
parser.add_argument('--gpu',               default='0',   type=str)
args = parser.parse_args()

DEVICE = args.gpu if args.gpu == 'cpu' else f'cuda:{args.gpu}'

DEFAULT_SIZE = {'yolov5n': 416, 'nanodet': 320, 'custom': 416}
MODEL_INPUT  = args.input_size if args.input_size else DEFAULT_SIZE[args.model]

os.makedirs(args.out_dir, exist_ok=True)


# ─── MODEL LOADERS ───────────────────────────────────────────
def load_yolov5n():
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend, AutoShape
    model = DetectMultiBackend(
        os.path.join('pretrained_models', 'yolov5n.pt'),
        device=torch.device(DEVICE), fuse=True)
    model = AutoShape(model)
    model.amp     = False
    model.conf    = args.conf
    model.classes = [0]
    return model


def load_custom(weight_path):
    """Loads a user-supplied YOLO-compatible .pt file using the same backend
    used for the built-in YOLOv5n demo model. Only Ultralytics YOLOv5/YOLOv8
    style weights are supported."""
    sys.path.insert(0, os.path.abspath('.'))
    from models.common import DetectMultiBackend, AutoShape
    model = DetectMultiBackend(
        weight_path, device=torch.device(DEVICE), fuse=True)
    model = AutoShape(model)
    model.amp     = False
    model.conf    = args.conf
    model.classes = [0]
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
    return model, cfg, pipeline


# ─── DETECTION ───────────────────────────────────────────────
def detect_yolov5(model, img_pil, conf_thr):
    img_np    = np.array(img_pil)
    orig_conf = model.conf
    model.conf = conf_thr
    with torch.no_grad():
        results = model(img_np, size=MODEL_INPUT)
    model.conf = orig_conf
    boxes = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().tolist():
        if int(cls) == 0:
            boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])
    return boxes


def detect_nanodet(model, nano_cfg, pipeline, img_pil, conf_thr):
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


# ─── DRAW BOXES ──────────────────────────────────────────────
def draw_boxes(pil_img, pred_boxes, label="Human"):
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2, conf) in pred_boxes:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1 - 15)), f'{label} {conf:.2f}', fill='red')
    return img


# ─── MAIN ────────────────────────────────────────────────────
def main():
    if args.model == 'custom' and not args.custom_model_path:
        print("[error] --custom_model_path is required when --model custom is used.")
        sys.exit(1)

    img_pil = Image.open(args.image).convert('RGB')
    fname   = os.path.splitext(os.path.basename(args.image))[0]

    print(f"[1] Loading model: {args.model} ...")
    nano_cfg = nano_pipeline = None
    if args.model == 'yolov5n':
        model = load_yolov5n()
    elif args.model == 'custom':
        model = load_custom(args.custom_model_path)
    elif args.model == 'nanodet':
        model, nano_cfg, nano_pipeline = load_nanodet()

    print(f"[2] Running detection on raw (uncloaked) frame: {args.image}")
    if args.model in ('yolov5n', 'custom'):
        pred_boxes = detect_yolov5(model, img_pil, args.conf)
    elif args.model == 'nanodet':
        pred_boxes = detect_nanodet(model, nano_cfg, nano_pipeline, img_pil, args.conf)

    vis = draw_boxes(img_pil, pred_boxes)
    out_path = os.path.join(args.out_dir, f'{fname}_aimbot_view.jpg')
    vis.convert('RGB').save(out_path)

    print(f"[3] Saved aimbot view -> {out_path}")
    if pred_boxes:
        confs = ", ".join(f"{c:.2f}" for *_, c in pred_boxes)
        print(f"    Before cloak: aimbot detected {len(pred_boxes)} target(s), confidence [{confs}]")
    else:
        print("    Before cloak: aimbot detected 0 targets on this frame.")


if __name__ == '__main__':
    main()