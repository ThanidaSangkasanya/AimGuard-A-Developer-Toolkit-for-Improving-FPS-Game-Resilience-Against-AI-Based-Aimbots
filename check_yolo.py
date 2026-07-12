import sys
import torch

sys.path.insert(0, '.')
from models.common import DetectMultiBackend
device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device_str}'
      + (f' ({torch.cuda.get_device_name(0)})' if device_str != 'cpu' else ' — no GPU detected!'))
model = DetectMultiBackend('pretrained_models/yolov5n.pt', device=torch.device(device_str))
print('✅ YOLOv5n loaded successfully:', type(model))
print('   Model is on device:', next(model.model.parameters()).device)