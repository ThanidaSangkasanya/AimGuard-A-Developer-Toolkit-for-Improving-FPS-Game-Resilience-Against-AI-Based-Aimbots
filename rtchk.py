import torch, sys, os 
from dotenv import load_dotenv; load_dotenv() 
from ultralytics import YOLO 
w = os.environ.get("RTDETR_WEIGHTS") 
m = YOLO(w).model.to("cuda:0").eval() 
x = torch.rand(1,3,640,640).cuda() 
out = m(x) 
print("type:", type(out)) 
print("is dict:", isinstance(out, dict)) 
raw = out[0] if isinstance(out,(list,tuple)) else out 
print("raw type:", type(raw)) 
print("raw shape:", raw.shape if hasattr(raw,"shape") else "no shape") 
