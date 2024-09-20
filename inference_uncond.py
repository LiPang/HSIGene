import sys
sys.path.append("./latentdiff")
sys.path.append("./models")
from os.path import join
import matplotlib.pyplot as plt
import utils.config as config
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
import cv2
import einops
import numpy as np
import os
from tqdm import tqdm
from scipy.io import *
import torch
from PIL import Image
import argparse

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, default=10)
parser.add_argument('--ddim-steps', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save-dir', type=str, default='save_uncond')
opts = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu

num_samples = opts.num_samples
ddim_steps = opts.ddim_steps
save_dir = opts.save_dir

H=256
W=256


resolution = 512
local_conditions = []
global_conditions = []
for i in range(1, 7):
    condition = np.zeros((resolution, resolution, 3))
    local_conditions.append(condition)

for i in range(7, 8):
    condition = np.zeros(768)
    global_conditions.append(condition)

metadata_emb=np.zeros((7))
global_maps=np.concatenate(global_conditions,axis=0)
detected_maps = np.concatenate(local_conditions, axis=2)
local_control = torch.from_numpy(detected_maps.copy()).float().cuda()
local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
global_control = torch.from_numpy(global_maps.copy()).float().cuda().clone()
global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)
metadata_control=torch.from_numpy(metadata_emb.copy()).float().cuda().clone().squeeze()

model = create_model('configs/inference.yaml').cpu()
state_dict = load_state_dict('checkpoints/last.ckpt', location='cpu')

model.load_state_dict(state_dict, strict=False)
model = model.cuda()
model = model.eval()

ddim_sampler = DDIMSampler(model)
if config.save_memory:
    model.low_vram_shift(is_diffusing=False)
cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([''] * num_samples)],
        'global_control': [global_control], "metadata": [metadata_control]}
shape = (4, H // 4, W // 4)
if config.save_memory:
    model.low_vram_shift(is_diffusing=True)


samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, metadata_control,conditioning=cond, verbose=False, eta=0.0,
                                                unconditional_guidance_scale=1,
                                                unconditional_conditioning=None, global_strength=1)
if config.save_memory:
    model.low_vram_shift(is_diffusing=False)
x_samples = model.decode_first_stage(samples)
x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c').cpu().numpy()
x_samples = x_samples * 0.5 + 0.5
results = [(x_samples[i]*255).clip(0, 255).astype(np.uint8)
           for i in range(num_samples)]


save_path = save_dir
os.makedirs(join(save_path, 'pngs'), exist_ok=True)
os.makedirs(join(save_path, 'mats'), exist_ok=True)


for i, image in enumerate(results):
    savemat(f'{save_path}/mats/hsi_{i}.mat', {'data': image})

    image_rgb = image[..., [20, 12, 4]]
    image_rgb = rsshow(image_rgb, 0.001)
    image_rgb = Image.fromarray((image_rgb*255).astype(np.uint8))
    image_rgb.save(f'{save_path}/pngs/hsi_{i}.png')

