# References:
# https://github.com/qubvel/segmentation_models.pytorch
import copy
import os
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as albu
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset as BaseDataset
from config import *
from torchsummaryX import summary


def setup_seed(seed):
    torch.manual_seed(seed)  # Set seeds for CPU
    torch.cuda.manual_seed_all(seed)  # Set seeds for all GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False  # Model convolutional layer pre-optimization off
    torch.backends.cudnn.deterministic = True  # Determined as the default convolution algorithm
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_csv(df):
    train_ids, test_ids = [], []
    for i, row in df.iterrows():
        obj_tag = row['obj_tag']
        if obj_tag == 'train':
            train_ids.append(row['img_name'])
        else:
            test_ids.append(row['img_name'])
    return train_ids, test_ids


def estimate_ratios(ood_ratio):
    avg_ratio = 0
    for str_code in [79, 158, 237]:
        setup_seed(str_code)
        df = pd.read_csv(os.path.join('k_folds', str(str_code) + 'Image_Patchs.csv'))

        train_ids, test_ids = read_csv(df)

        images_dir, masks_dir = 'Image_Patchs', 'Mask_Patchs'
        images_ood_dir, masks_ood_dir = 'OoD_Patchs', 'Mask_Patchs'
        net, ENCODER, ENCODER_WEIGHTS, CLASSES, ACTIVATION, model = get_config()

        df_ood = pd.read_csv(
            os.path.join('HardOoD', str(str_code) + '_' + net + '_' + ENCODER + '_' + "HardOoD.csv"))
        ood_ids = []
        for i, row in df_ood.iterrows():
            ood_ids.append(row['hard_ood'])

        random.shuffle(ood_ids)
        ood_ids = ood_ids[:round(len(ood_ids) * ood_ratio)]

        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        train_dataset = Dataset(images_dir, masks_dir, train_ids, classes=CLASSES,
                                augmentation=get_train_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn))
        ood_dataset = OoD_Dataset(images_ood_dir, masks_ood_dir, ood_ids, classes=CLASSES,
                                  preprocessing=get_preprocessing(preprocessing_fn))

        combine_dataset = ConcatDataset([train_dataset, ood_dataset])
        pos_samples = 0
        neg_samples = 0
        for i in range(len(combine_dataset)):
            image, gt_mask = combine_dataset[i]
            fm_mean = np.mean(gt_mask[0, ...])
            if fm_mean == 0:
                pos_samples = pos_samples + 1
            elif 0 < fm_mean < 1:
                pos_samples = pos_samples + 1
                neg_samples = neg_samples + 1
            elif fm_mean == 1:
                neg_samples = neg_samples + 1
        pos_neg_ratio = float(pos_samples) / float(neg_samples)
        # print('positive and negative ratio: ', pos_neg_ratio)
        avg_ratio = avg_ratio + pos_neg_ratio
    avg_ratio = float(avg_ratio) / 3
    # print('Average postive and negative ratio: ', avg_ratio)
    return avg_ratio


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def img2tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=img2tensor, mask=img2tensor),
    ]
    return albu.Compose(_transform)


def get_train_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.RandomCrop(height=128, width=128, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

    ]

    return albu.Compose(train_transform)


class Dataset(BaseDataset):
    CLASSES = ['0', '1', '2', '3', '4', '5', '6']  # 7 classes: background + 6 classes

    def __init__(
            self,
            images_dir,
            masks_dir,
            ids,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], -1)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class OoD_Dataset(BaseDataset):
    CLASSES = ['0', '1', '2', '3', '4', '5', '6']  # 7 classes: background + 6 classes

    def __init__(
            self,
            images_dir,
            masks_dir,
            ids,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
