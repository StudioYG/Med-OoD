import cv2
import numpy as np
from imutils import paths
from tqdm import tqdm


def img2tiles(img, imh, imw, patch_sz):
    padh = patch_sz - imh % patch_sz
    padw = patch_sz - imw % patch_sz
    img = np.pad(img, [[padh // 2, padh - padh // 2], [padw // 2, padw - padw // 2], [0, 0]],
                 constant_values=0)
    img = img.reshape((imh + padh) // patch_sz, patch_sz, (imw + padw) // patch_sz, patch_sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, patch_sz, patch_sz, 3)

    return img


def tiles2img(img, imh, imw, patch_sz):
    padh = patch_sz - imh % patch_sz
    padw = patch_sz - imw % patch_sz
    img = img.reshape((imh + padh) // patch_sz, (imw + padw) // patch_sz, patch_sz, patch_sz, 3).transpose(0, 2, 1, 3,
                                                                                                           4)
    img = img.reshape(imh + padh, imw + padw, 3)
    img = img[padh // 2:-(padh - padh // 2), padw // 2:-(padw - padw // 2)]
    return img


def mask2tiles(mask, imh, imw, patch_sz):
    padh = patch_sz - imh % patch_sz
    padw = patch_sz - imw % patch_sz

    mask = np.pad(mask, [[padh // 2, padh - padh // 2], [padw // 2, padw - padw // 2]],
                  constant_values=0)
    mask = mask.reshape((imh + padh) // patch_sz, patch_sz, (imw + padw) // patch_sz, patch_sz)
    mask = mask.transpose(0, 2, 1, 3).reshape(-1, patch_sz, patch_sz)

    return mask


def tiles2mask(mask, imh, imw, patch_sz):
    padh = patch_sz - imh % patch_sz
    padw = patch_sz - imw % patch_sz
    mask = mask.reshape((imh + padh) // patch_sz, (imw + padw) // patch_sz, patch_sz, patch_sz).transpose(0, 2, 1, 3)
    mask = mask.reshape(imh + padh, imw + padw)
    mask = mask[padh // 2:-(padh - padh // 2), padw // 2:-(padw - padw // 2)]
    return mask


patch_sz = 128
for img_path in tqdm(paths.list_images('Lizard_Images')):
    img = cv2.imread(img_path)
    mask = cv2.imread(img_path.replace('Lizard_Images', 'Semantic_Labels'), -1)
    imh, imw = img.shape[0], img.shape[1]
    padh = patch_sz - imh % patch_sz
    padw = patch_sz - imw % patch_sz

    img_list = img2tiles(img, imh, imw, patch_sz)
    mask_list = mask2tiles(mask, imh, imw, patch_sz)

    for i, (im, ma) in enumerate(zip(img_list, mask_list)):
        cv2.imwrite(img_path.replace('Lizard_Images', 'Image_Patchs').replace('.png', '_'+str(i)+'.png'), im)
        cv2.imwrite(img_path.replace('Lizard_Images', 'Mask_Patchs').replace('.png', '_' + str(i) + '.png'), ma)


