import sys, os, torch

os.environ['NANODET_ROOT'] = 'nanodet'
sys.path.insert(0, os.environ['NANODET_ROOT'])

from nanodet.util import cfg, load_config, Logger as NL
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight

load_config(cfg, os.path.join(os.environ['NANODET_ROOT'], 'config', 'nanodet-plus-m_320.yml'))
model = build_model(cfg.model)
ckpt = torch.load(os.path.join(os.environ['NANODET_ROOT'], 'nanodet', 'nanodet-plus-m_320.pth'), map_location='cpu')
load_model_weight(model, ckpt, NL(0, use_tensorboard=False))

print('✅ NanoDet-Plus loaded successfully:', type(model))