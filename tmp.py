import torch.nn.functional as F
import torch
import numpy as np
from torchvision.transforms import ToPILImage
from src.models.ShadeNet import ShadeNet

dummy = torch.zeros((1, 2, 3, 4)).cuda()
device = dummy.device

frame_shapes = [
    (1, 3, 528, 960),
    (1, 3, 528, 960),
    (1, 3, 528, 960),
    (1, 3, 528, 960),
    (1, 2, 528, 960),
    (1, 2, 528, 960),
    (1, 2, 528, 960),
]


model = ShadeNet().to(device)

g_state, d_state = model.beginState(dummy.device, 960, 528)

inputs = [torch.zeros(shape).to(device) for shape in frame_shapes]

model.eval()

res, g_s, d_s, smv, smv_resi = model(
    inputs[0],
    inputs[1],
    inputs[2],
    inputs[3],
    inputs[4],
    inputs[5],
    inputs[6],
    g_state,
    d_state,
)

g_state = [g_s, g_state[1], g_state[2]]
d_state = [d_s, d_state[1], d_state[2]]

res, g_s, d_s, smv, smv_resi = model(
    inputs[0],
    inputs[1],
    inputs[2],
    inputs[3],
    inputs[4],
    inputs[5],
    inputs[6],
    g_state,
    d_state,
)