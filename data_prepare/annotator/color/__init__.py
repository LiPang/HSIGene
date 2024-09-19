from typing import Any
import cv2
import numpy as np

class ColorDetector:
    def __call__(self, img):
        # img_np = np.array(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean = hsv[:,:,0].mean()
        s_mean = hsv[:,:,1].mean()
        v_mean = hsv[:,:,2].mean()
        return h_mean, s_mean, v_mean