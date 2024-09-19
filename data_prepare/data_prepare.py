# coding=utf-8
import os
import sys
import cv2
import matplotlib.pyplot as plt

import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.sketch import SketchDetector
from annotator.uniformer import SAMDetector


def rsshow(I, scale=0.005):
	low, high = np.quantile(I, [scale, 1 - scale])
	I[I > high] = high
	I[I < low] = low
	I = (I - low) / (high - low)
	return I


hed_annotator = HEDdetector()
sam_annotator = SAMDetector()
sketch_annotator = SketchDetector()
mlsd_annotator = MLSDdetector()


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='candidates')
parser.add_argument('--savedir', type=str, default='conditions')
parser.add_argument('--conditions', type=str, default='all',
					choices=['all', 'hed', 'segmentation', 'sketch', 'mlsd', 'content'])
opts = parser.parse_args()


data_dir = opts.datadir
fns = [t for t in os.listdir(data_dir)
	   if t.endswith('.png') or t.endswith('.jpg')]

save_dir = opts.savedir
os.makedirs(save_dir, exist_ok=True)


if opts.conditions == 'all':
	conditions_names = ['hed', 'segmentation', 'sketch', 'mlsd', 'content']
else:
	conditions_names = [opts.conditions]

for fn in tqdm(fns):
	fn_path = os.path.join(data_dir, fn)
	image = cv2.imread(fn_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = rsshow(np.array(image), 0)
	image = (image * 255).astype(np.uint8)

	save_path = os.path.join(save_dir, fn.split('.')[0])
	os.makedirs(save_path, exist_ok=True)

	for condition_name in conditions_names:
		if condition_name == 'hed':
			condition = hed_annotator(image)
		elif condition_name == 'segmentation':
			image_seg, seg_labels = sam_annotator(image)
			image_segment_with_color = np.zeros_like(image)
			seg_labels = seg_labels[0].astype(np.uint8)
			for i in range(1, seg_labels.max() + 1):
				mask = seg_labels == i
				image_segment_with_color[mask] = image[mask].mean(0)
			condition = image_segment_with_color
		elif condition_name == 'sketch':
			condition = sketch_annotator(image)
		elif condition_name == 'mlsd':
			condition = mlsd_annotator(image, 0.05, 20)
		elif condition_name == 'content':
			condition = image
		else:
			raise ValueError('Invalid Condition')


		Image.fromarray(condition).save(
			os.path.join(save_path, condition_name+'.png'))

