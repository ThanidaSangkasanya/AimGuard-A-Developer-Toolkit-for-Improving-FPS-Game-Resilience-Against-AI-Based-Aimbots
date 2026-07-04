import torch, sys 
sys.path.insert(0, ".") 
from models.common import DetectMultiBackend 
print("Import OK", flush=True) 
m = DetectMultiBackend("pretrained_models/yolov5n.pt", device=torch.device("cuda:0"), fuse=True) 
print("DetectMultiBackend OK", flush=True) 
