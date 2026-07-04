import torch 
print("CUDA:", torch.cuda.is_available()) 
ckpt = torch.load("pretrained_models/yolov5n.pt", map_location="cpu", weights_only=False) 
print("Loaded keys:", list(ckpt.keys())) 
