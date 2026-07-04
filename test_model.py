import torch, sys
sys.path.insert(0, '.') 
from models.common import DetectMultiBackend 
model = DetectMultiBackend('pretrained_models/yolov5n.pt', device=torch.device('cpu')) 
print('Model loaded OK') 
