import os
import cv2
import shutil
import torch
import numpy as np
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.transforms import RandomHorizontalFlip


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MPDD(Dataset):
    
    CLASS_NAMES = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    
    def __init__(self, 
                 root: str,
                 class_name: str,
                 train: bool = True,
                 normalize: str = 'imagebind',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size'), kwargs.get('crp_size')]
        
        # load dataset
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
        elif self.class_name is None:  # load all classes
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data(self.class_name)
        
        # set transforms
        if normalize == "imagebind":
            self.transform = T.Compose(  # for imagebind
                [
                    T.Resize(
                        224, interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
                T.CenterCrop(kwargs.get('crp_size', 224)),
                T.ToTensor(),
                T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])])
            
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('img_size'), Image.NEAREST),
            T.CenterCrop(kwargs.get('crp_size')),
            T.ToTensor()])
        
        self.class_to_idx = {'bracket_black': 0, 'bracket_brown': 1, 'bracket_white': 2,
                             'connector': 3, 'metal_plate': 4, 'tubes': 5}
        self.idx_to_class = {0: 'bracket_black', 1: 'bracket_brown', 2: 'bracket_white',
                             3: 'connector', 4: 'metal_plate', 5: 'tubes'}

    def __getitem__(self, idx):
        image_path, label, mask, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        image = self.transform(image)
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
                
        if self.train:
            label = self.class_to_idx[class_name]
        
        return image, label, mask, class_name

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self, class_name):
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths = [], [], []

        img_dir = os.path.join(self.root, class_name, phase)
        gt_dir = os.path.join(self.root, class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
        class_names = [class_name] * len(image_paths)
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
    
    def update_class_to_idx(self, class_to_idx):
        for class_name in self.class_to_idx.keys():
            self.class_to_idx[class_name] = class_to_idx[class_name]
        class_names = self.class_to_idx.keys()
        idxs = self.class_to_idx.values()
        self.idx_to_class = dict(zip(idxs, class_names))

            