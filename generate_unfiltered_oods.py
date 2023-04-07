import os
import cv2
import numpy as np
from tqdm import tqdm
from toolkit import visualize

for file in tqdm(os.listdir(r'Image_Patchs')):
    img = cv2.imread(os.path.join('Image_Patchs', file))
    mask = cv2.imread(os.path.join('Mask_Patchs', file), -1)
    masks = [(mask == 0)] * img.shape[2]
    mask = np.stack(masks, axis=-1).astype('uint8')
    ood_img = np.multiply(img, mask)
    # visualize(ood_img=ood_img)
    cv2.imwrite(os.path.join('OoD_Patchs', file), ood_img)
