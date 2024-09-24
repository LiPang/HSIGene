import os
import numpy as np
from annotator.uniformer.mmseg.apis import init_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette
from annotator.util import annotator_ckpts_path



class SAMDetector:
    def __init__(self):
        from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
        remote_model_path = "https://huggingface.co/ybelkada/segment-anything/tree/main/checkpoints/sam_vit_h_4b8939.pth"
        modelpath = os.path.join(annotator_ckpts_path, 'sam_vit_h_4b8939.pth')

        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)

        sam = sam_model_registry["default"](checkpoint=modelpath)
        sam.to(device='cuda')
        mask_generator = SamAutomaticMaskGenerator(sam)
        self.mask_generator = mask_generator

        config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "uniformer", "exp", "upernet_global_small",
                                   "config.py")
        modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
        self.model = init_segmentor(config_file, modelpath).cuda()

    def __call__(self, img):
        masks = self.mask_generator.generate(img)
        result = np.zeros((1, img.shape[0], img.shape[1]))
        for i, mask in enumerate(masks):
            mask = mask['segmentation']
            result[0, mask] = i + 1
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img, result

