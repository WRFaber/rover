import torch
from mars_policy import PolicyNet
import numpy as np


model = PolicyNet()
model.load_state_dict(torch.load("policy_net.pth", map_location=torch.device("cpu")))
model.eval()


quantized_params = {}
scales = {}

for name, param in model.named_parameters():
    arr = param.detach().numpy()
    scale = np.max(np.abs(arr)) / 127 if np.max(np.abs(arr)) > 0 else 1.0
    quantized = np.round(arr / scale).astype(np.int8)
    quantized_params[name] = quantized
    scales[name] = scale


with open("policynet_quantized.py", "w") as f:
    for name, array in quantized_params.items():
        f.write(f"{name.replace('.', '_')} = {array.tolist()}\n")
    for name, scale in scales.items():
        f.write(f"{name.replace('.', '_')}_scale = {scale}\n")