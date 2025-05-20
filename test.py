import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from validate import validate
from datasets.mvtec import MVTEC, MVTECANO
from datasets.ksdd import KSDD
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS
from datasets.ReinADtest import SELFDATA, SELFMVTEC
from functools import reduce
from operator import add
from models.learner import VATLearner
from utils import init_seeds, load_mc_reference_features
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS, MVTEC_TO_KSDD, MVTEC_TO_ReinADtest, ReinADtrain_TO_ReinADtest

import os


warnings.filterwarnings('ignore')
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS, 'mvtec_to_ksdd': MVTEC_TO_KSDD,
            'mvtec_to_ReinADtest': MVTEC_TO_ReinADtest, 'ReinADtrain_to_ReinADtest':ReinADtrain_TO_ReinADtest}

def main(args):
    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]

    checkpoint_path = 'checkpoints/checkpoint.pth' 


    backbone = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True).eval() 
    backbone = backbone.to(args.device)
    feat_ids = list(range(3, 17))
    nbottlenecks = [3, 4, 6, 3] 
    bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
    lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
    stack_ids = torch.tensor(lids).bincount().__reversed__().cumsum(dim=0)[:3]
    hpn_learner = VATLearner(inch = [x for x in reversed(nbottlenecks[-3:])],shot=args.num_ref_shot)
    hpn_learner.to(args.device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        hpn_learner.load_state_dict(checkpoint['hpn_learner'])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")

    s_res = []
    test_ref_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, args.num_ref_shot)
    for class_name in CLASSES['unseen']:
        '''
        if class_name in MVTEC.CLASS_NAMES:
            test_dataset = MVTEC(args.test_dataset_dir, class_name=class_name, train=False,
                                    normalize='w50',
                                    img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        '''
        if class_name in VISA.CLASS_NAMES:
            test_dataset = VISA(args.test_dataset_dir, class_name=class_name, train=False,
                                normalize='w50',
                                img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in BTAD.CLASS_NAMES:
            test_dataset = BTAD(args.test_dataset_dir, class_name=class_name, train=False,
                                normalize='w50',
                                img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in MVTEC3D.CLASS_NAMES:
            test_dataset = MVTEC3D(args.test_dataset_dir, class_name=class_name, train=False,
                                    normalize='w50',
                                    img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in MPDD.CLASS_NAMES:
            test_dataset = MPDD(args.test_dataset_dir, class_name=class_name, train=False,
                                normalize='w50',
                                img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in MVTECLOCO.CLASS_NAMES:
            test_dataset = MVTECLOCO(args.test_dataset_dir, class_name=class_name, train=False,
                                normalize='w50',
                                img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in BRATS.CLASS_NAMES:
            test_dataset = BRATS(args.test_dataset_dir, class_name=class_name, train=False,
                                    normalize='w50',
                                    img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in KSDD.CLASS_NAMES:
            test_dataset = KSDD(args.test_dataset_dir, class_name=class_name, train=False,
                                    normalize='w50',
                                    img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        elif class_name in SELFDATA.CLASS_NAMES:
            #test_dataset = SELFMVTEC(args.test_dataset_dir, class_name=class_name, train=False,
            #                        normalize='w50',
            #                        img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
            test_dataset = SELFDATA(args.test_dataset_dir, class_name=class_name, train=False,
                                    normalize='w50',
                                    img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)                
        else:
            raise ValueError('Unrecognized class name: {}'.format(class_name))
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
        )

        metrics = validate(args, backbone, hpn_learner, test_loader, test_ref_features[class_name], args.device, feat_ids, bottleneck_ids, lids, stack_ids, args.num_ref_shot)
        img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics['scores']
        
        print("Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
            class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
        s_res.append(metrics['scores'])
        
    s_res = np.array(s_res)
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(s_res, axis=0)
    print('(Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
        img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="visa_to_mvtec")
    parser.add_argument('--test_dataset_dir', type=str, default="")
    parser.add_argument('--test_ref_feature_dir', type=str, default="./ref_features/w50/mvtec_1shot")
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    parser.add_argument("--num_ref_shot", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    init_seeds(42)
    
    main(args)
