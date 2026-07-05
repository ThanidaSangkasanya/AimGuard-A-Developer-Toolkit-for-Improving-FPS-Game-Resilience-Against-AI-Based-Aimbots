import torch, sys, os 
from dotenv import load_dotenv; load_dotenv() 
from ultralytics import YOLO 
w = os.environ.get("RTDETR_WEIGHTS") 
m = YOLO(w).model.to("cuda:0").eval() 
x = torch.rand(1,3,640,640).cuda() 
out = m(x) 
raw = out[0] if isinstance(out,(tuple,list)) else out 
print("idx4 range:", raw[...,4].min().item(), raw[...,4].max().item()) 
print("idx5 range:", raw[...,5].min().item(), raw[...,5].max().item()) 
print("idx0 range:", raw[...,0].min().item(), raw[...,0].max().item()) 
