import os
import random
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T

import cv2
import glob
import imgaug.augmenters as iaa
import albumentations as A


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



class SELFTRAINDATA(Dataset):

    #CLASS_NAMES = ['PCB_cable_crush_304_400', 'PCB_cable_scratch_304_400', 'PCB_solder_joint_side', 'PCB_solder_joint_top', 'PCB_terminal', 'cable', 'chaxuansi', 'fanguangtie', 'led_1_1368_1216', 
    #           'led_1_608_608', 'led_2_1368_1216', 'led_2_608_608', 'lens_2', 'pinline', 'plastic_box', 'plastic_cover_1', 'solder_1', 'x_ray_1', 'x_ray_2', 'x_ray_3', 'x_ray_4', 'x_ray_5', 'x_ray_6', 
    #           'solder_2', 'solder_3', 'x_ray_10', 'x_ray_11', 'x_ray_12', 'x_ray_7', 'x_ray_8', 'x_ray_9']
    CLASS_NAMES = ['PCB_terminal', 'bearing_10', 'bearing_11', 'bearing_12', 'bearing_2', 'bearing_3', 'bearing_4', 'bearing_5', 'bearing_6', 'bearing_7', 'bearing_8', 'bearing_9', 'cable_2', 'led_2', 'lens_1', 'miniled_1', 'miniled_2', 'miniled_3', 'motor_base_10', 'motor_base_11', 'motor_base_3', 'motor_base_6', 'motor_base_9', 'pinline', 'piston_ring_2', 'plastic_cover_2', 'plastic_cover_3', 'plastic_cover_4', 'plastic_cover_5', 'plastic_cover_6', 'profile_surface_2', 'reflective_sheet', 'round_tube', 'solder_1', 'solder_2', 'solder_3', 'suspension_wire']
    def __init__(self, 
                 root: str,
                 class_name: str = 'PCB_cable_crush_304_1200', 
                 train: bool = True,
                 normalize: str = 'imagebind',
                 **kwargs) -> None:
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
        elif self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data(self.class_name)
        
        if normalize == "imagebind":
            self.transform = T.Compose([ 
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
                T.CenterCrop(kwargs.get('crp_size', 224)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 224), Image.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label, mask_path, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        img, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        return img, label, mask, class_name

    def _load_image_and_mask(self, image_path, label, mask_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.target_transform(mask)
        
        return img, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths = [], [], []
        phase = 'test' 
        
        image_dir = os.path.join(self.root, class_name, 'Data', 'Images')
        mask_dir = os.path.join(self.root, class_name, 'Data', 'Vis', 'Masks_255')

        normal_image_dir = os.path.join(image_dir, 'Normal')
        anomaly_image_dir = os.path.join(image_dir, 'Anomaly')

        normal_images = sorted([os.path.join(normal_image_dir, f) for f in os.listdir(normal_image_dir) if f.endswith('.jpg')])
        anomaly_images = sorted([os.path.join(anomaly_image_dir, f) for f in os.listdir(anomaly_image_dir) if f.endswith('.jpg')])

        normal_masks = [None] * len(normal_images)
        anomaly_masks = sorted([os.path.join(mask_dir, os.path.splitext(os.path.basename(f))[0] + '.png') for f in anomaly_images])

        image_paths.extend(normal_images + anomaly_images)
        mask_paths.extend(normal_masks + anomaly_masks)
        labels.extend([0] * len(normal_images) + [1] * len(anomaly_images))
        class_names = [class_name] * (len(normal_images) + len(anomaly_images))

        return image_paths, labels, mask_paths, class_names

    def _load_all_data(self, class_names=None):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_class_names = []
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            image_paths, labels, mask_paths, class_names = self._load_data(class_name)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_class_names.extend(class_names)
        return all_image_paths, all_labels, all_mask_paths, all_class_names


def get_normal_image_paths_mvtec(root, class_name):
    phase = 'test' 
    image_paths = []

    image_dir = os.path.join(root, class_name, 'Data', 'Images')
    normal_image_dir = os.path.join(image_dir, 'Normal')

    normal_images = sorted([os.path.join(normal_image_dir, f) for f in os.listdir(normal_image_dir) if f.endswith('.jpg')])

    image_paths.extend(normal_images)

    return image_paths
