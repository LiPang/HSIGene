import random
import sys
sys.path.append("./latentdiff")
sys.path.append("./models")
from os.path import join
from scipy.io import loadmat
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
from data_prepare.annotator.content import ContentDetector
import argparse

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I

parser = argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, default=5)
parser.add_argument('--ddim-steps', type=int, default=50)
parser.add_argument('--global-strength', type=float, default=1.0)
parser.add_argument('--conditions', type=str, default='', help='hed, segmentation, sketch, mlsd, content, text')
parser.add_argument('--prompt', type=str, default='', choices=['Farmland', 'Architecture', 'City Building', 'Wasteland'])
parser.add_argument('--condition-dir', type=str, default='data_prepare/conditions')
parser.add_argument('--fns', type=str, default='samples')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save-dir', type=str, default='save')
opts = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
seed_everywhere(1234)

fns = opts.fns
num_samples = opts.num_samples
ddim_steps = opts.ddim_steps
global_strength = opts.global_strength
prompt = opts.prompt
conditions = opts.conditions
condition_dir = opts.condition_dir
save_dir = opts.save_dir

H=256
W=256


conditions_names = conditions.split(' ')
number_cond_dict = {'hed': 1, 'segmentation':3, 'sketch':5, 'mlsd':6, 'content':7, 'text':0}
cond_number_dict = {1:'hed', 3:'segmentation', 5:'sketch', 6:'mlsd', 7:'content', 0:'text'}
number_cond = [number_cond_dict[t] for t in conditions_names]

resolution = 512
local_conditions = []
global_conditions = []
for i in range(1, 7):
    if i in number_cond:
        fn_path = join(condition_dir, fns, cond_number_dict[i]+'.png')
        condition = cv2.imread(fn_path)
        condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
        condition = condition.astype(np.float32) / 255.0
        condition = cv2.resize(condition, (resolution, resolution))
    else:
        condition = np.zeros((resolution, resolution, 3))
    local_conditions.append(condition)



for i in range(7, 8):
    if i in number_cond:
        content_model = ContentDetector('data_prepare/annotator/ckpts')
        fn_path = join(condition_dir, fns, cond_number_dict[i]+'.png')
        condition = cv2.imread(fn_path)
        condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
        condition = rsshow(condition, 0)
        condition = (condition * 255).astype(np.uint8)
        condition = content_model(condition)
    else:
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

model.load_state_dict(state_dict, strict=True)
model = model.cuda()
model = model.eval()

ddim_sampler = DDIMSampler(model)
if config.save_memory:
    model.low_vram_shift(is_diffusing=False)
cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)],
        'global_control': [global_control], "metadata": [metadata_control]}
shape = (4, H // 4, W // 4)
if config.save_memory:
    model.low_vram_shift(is_diffusing=True)


## add x_T
samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, metadata_control,conditioning=cond, verbose=False, eta=0.0,
                                                unconditional_guidance_scale=1,
                                                unconditional_conditioning=None, global_strength=global_strength)
if config.save_memory:
    model.low_vram_shift(is_diffusing=False)
x_samples = model.decode_first_stage(samples)
x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c').cpu().numpy()
x_samples = x_samples * 0.5 + 0.5
results = [(x_samples[i]*255).clip(0, 255).astype(np.uint8)
           for i in range(num_samples)]


save_path = os.path.join(save_dir, fns, '_'.join(conditions_names))
os.makedirs(join(save_path, 'pngs'), exist_ok=True)
os.makedirs(join(save_path, 'mats'), exist_ok=True)
os.makedirs(join(save_path, 'conds'), exist_ok=True)

for i, image in enumerate(results):
    # save mat
    savemat(f'{save_path}/mats/f{i}.mat', {'data': image})

    # save png
    image_rgb = image[..., [20, 12, 4]]
    image_rgb = rsshow(image_rgb)
    image_rgb = Image.fromarray((image_rgb*255).astype(np.uint8))
    image_rgb.save(f'{save_path}/pngs/f{i}.png')

# save conditions
for i in range(0, local_control.shape[1], 3):
    image = local_control[0, i:i+3].cpu().numpy().transpose((1, 2, 0))
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(f'{save_path}/conds/cond{i//3}.png')
