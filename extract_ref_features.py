import os
import argparse
import numpy as np
from PIL import Image

import torch
import tqdm
import timm
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.ReinADtest import SELFDATA
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS
from datasets.ksdd import KSDD
from functools import reduce
from operator import add


class FEWSHOTDATA(Dataset):
    
    def __init__(self, 
                 root: str,
                 class_name: str = 'bottle', 
                 train: bool = True,
                 **kwargs) -> None:
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.mask_size = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)

        # set transforms
        self.transform = T.Compose([
            T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
            T.CenterCrop(kwargs.get('crp_size', 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 256), T.InterpolationMode.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 256)),
            T.ToTensor()])
    
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
            mask = torch.zeros([1, self.mask_size[0], self.mask_size[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        
        return img, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths = [], [], []
        phase = 'train' if self.train else 'test'
        
        image_dir = os.path.join(self.root, class_name, phase)
        mask_dir = os.path.join(self.root, class_name, 'ground_truth')
        print(image_dir)
        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
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
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                    
        class_names = [class_name] * len(image_paths)
        
        return image_paths, labels, mask_paths, class_names
    

SETTINGS = {'mvtec': MVTEC.CLASS_NAMES, 'visa': VISA.CLASS_NAMES,
            'btad': BTAD.CLASS_NAMES, 'mvtec3d': MVTEC3D.CLASS_NAMES,
            'mpdd': MPDD.CLASS_NAMES, 'mvtecloco': MVTECLOCO.CLASS_NAMES,
            'brats': BRATS.CLASS_NAMES, 'ksdd': KSDD.CLASS_NAMES,
            'reinad': SELFDATA.CLASS_NAMES}


def main(args):
    image_size = 512
    device = 'cuda:0'
    root_dir = args.few_shot_dir
    
    backbone = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True).eval()  # the pretrained checkpoint will be in /home/.cache/torch/hub/checkpoints/
    backbone = backbone.to(device)
    
    feat_ids = list(range(3, 17))  
    nbottlenecks = [3, 4, 6, 3]   
    bottleneck_ids = reduce(add, [list(range(x)) for x in nbottlenecks])
    lids = reduce(add, [[i+1]*x for i, x in enumerate(nbottlenecks)])
    
    if args.dataset in SETTINGS.keys():
        CLASS_NAMES = SETTINGS[args.dataset]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.dataset}.")

    for class_name in CLASS_NAMES:
        train_dataset = FEWSHOTDATA(root_dir, class_name=class_name, train=True, 
                                 img_size=image_size, crp_size=image_size,
                                 msk_size=image_size, msk_crp_size=image_size)

        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False
        )
        
        features = [[] for _ in range(14)]

        for batch in tqdm.tqdm(train_loader):
            images, _, _, _ = batch
            with torch.no_grad():
                feats = extract_feat_res(images.to(device), 
                                      backbone, 
                                      feat_ids, 
                                      bottleneck_ids, 
                                      lids)
            for i in range(14):
                features[i].append(feats[i].cpu()) 

        os.makedirs(os.path.join(args.save_dir, class_name), exist_ok=True)
        
        for layer_idx in range(14):
            layer_feats = torch.cat(features[layer_idx], dim=0) 
            
            np.save(
                os.path.join(args.save_dir, class_name, f'layer{layer_idx+1}.npy'),
                layer_feats.numpy() 
            )

def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    feats = []
    
    # Layer 0
    feat = backbone.conv1(img)
    feat = backbone.bn1(feat)
    feat = backbone.act1.forward(feat)
    feat = backbone.maxpool(feat)
    
    # Layers 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__(f'layer{lid}')[bid].conv1(feat)
        feat = backbone.__getattr__(f'layer{lid}')[bid].bn1(feat)
        feat = backbone.__getattr__(f'layer{lid}')[bid].act1(feat)
        
        feat = backbone.__getattr__(f'layer{lid}')[bid].conv2(feat)
        feat = backbone.__getattr__(f'layer{lid}')[bid].bn2(feat)
        feat = backbone.__getattr__(f'layer{lid}')[bid].act2(feat)
        
        feat = backbone.__getattr__(f'layer{lid}')[bid].conv3(feat)
        feat = backbone.__getattr__(f'layer{lid}')[bid].bn3(feat)
        
        if bid == 0:
            res = backbone.__getattr__(f'layer{lid}')[bid].downsample(res)
            
        feat += res
        feat = backbone.__getattr__(f'layer{lid}')[bid].act3(feat)
        if hid + 1 in feat_ids:
            feats.append(feat.clone())
        
    return feats
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="reinad")
    parser.add_argument('--few_shot_dir', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="")
    args = parser.parse_args()
    main(args)