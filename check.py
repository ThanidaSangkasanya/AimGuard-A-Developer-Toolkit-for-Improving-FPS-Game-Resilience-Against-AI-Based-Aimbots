try:
    import sys
    sys.path.insert(0, '.')
    from models.yolo import DetectionModel
    print("YOLOv5: OK")
except Exception as e:
    print(f"YOLOv5: FAIL — {e}")

try:
    from nanodet.model.arch import build_model
    print("NanoDet: OK")
except Exception as e:
    print(f"NanoDet: FAIL — {e}")

try:
    from ultralytics import RTDETR
    print("RT-DETR: OK")
except Exception as e:
    print(f"RT-DETR: FAIL — {e}")