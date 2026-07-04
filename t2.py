import torch, sys 
sys.path.insert(0, ".") 
from models.experimental import attempt_load 
print("Importing OK", flush=True) 
m = attempt_load("pretrained_models/yolov5n.pt", device="cpu") 
print("CPU load OK", flush=True) 
m = m.to("cuda:0") 
print("GPU move OK", flush=True) 
