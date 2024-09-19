import os
import cv2
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, CLIPModel



class ContentDetector:
    def __init__(self, annotator_ckpts_path=None):
        if annotator_ckpts_path is None:
            from annotator.util import annotator_ckpts_path
        model_name = os.path.join(annotator_ckpts_path, 'clip/checkpoint-2500')
        model_name_un= os.path.join(annotator_ckpts_path, 'clip/clip-vit-large-patch14')

        model = CLIPModel.from_pretrained(model_name, cache_dir=annotator_ckpts_path).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_name_un, cache_dir=annotator_ckpts_path)

        # model.load_state_dict(checkpoint['state_dict'])
        self.model=model

    def __call__(self, img):
        assert img.ndim == 3
        with torch.no_grad():
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=img, return_tensors="pt").to('cuda')
            image_features = self.model.get_image_features(**inputs)
            image_feature = image_features[0].detach().cpu().numpy()
        return image_feature
