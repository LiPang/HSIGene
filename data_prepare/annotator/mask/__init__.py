import cv2
import numpy as np
import random
class MaskDetector:
    def __call__(self,image,train=True):
        def add_mask_to_image(image):
            # 生成与原始图像大小相同的二值掩码
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # 确定掩盖区域的范围
            min_width=image.shape[0] //6
            max_width=image.shape[0] //2
            width=random.randint(min_width,max_width)
            start_width=random.randint(0,image.shape[0]-width)
            start_height=random.randint(0,image.shape[0]-width)
            # start_row = image.shape[0] // 5
            # end_row = image.shape[0] // 3
            # print(start_width,start_height,width)
            
            # 在掩码中将对应区域置为1
            mask[start_width:start_width+width, start_height:start_height+width] = 1
            
            # 将原始图像中对应掩码区域的像素值置为0
            masked_image = np.copy(image)
            masked_image[mask == 1] = 0
            
            # 将二值掩码扩展为三通道，与原始图像的通道数相同
            mask_expanded = np.expand_dims(mask, axis=2)
            # mask_expanded = np.repeat(mask_expanded, 3, axis=2)
            # 
            # 将原始图像和掩码合并成一个四通道图像
            merged_image = np.concatenate((masked_image, mask_expanded), axis=2)
            
            return merged_image

        # 读取原始图像
        # image_path = '/home/tdt/snap/tdt/txt2img/diffuser_model/dataset/RSICD/RSICD_images/RSICD_images/00001.jpg'
        # image = cv2.imread(image_path)

        # 调用函数添加掩码
        image = np.array(image)
        if train==False:
           image = np.resize(image, (512, 512, image.shape[2]))
        result_image = add_mask_to_image(image)

        # # 保存合并后的图像
        # output_path = 'path_to_save_result_image.jpg'
        # cv2.imwrite(output_path, result_image)
        return result_image
