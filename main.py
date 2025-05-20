import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import timm
import torch.nn.functional as F
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
from datasets.ReinADtest import SELFDATA
from datasets.ReinADtrain import SELFTRAINDATA
from functools import reduce
from operator import add
from models.learner import VATLearner
from utils import init_seeds, get_mc_matched_ref_features, get_mc_reference_features, multilayer_correlation ,extract_feat_res, stack_feats, load_mc_reference_features
from losses.loss import BinaryFocalLoss
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS, MVTEC_TO_KSDD, MVTEC_TO_ReinADtest, ReinADtrain_TO_ReinADtest
import os

warnings.filterwarnings('ignore')

SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS, 'mvtec_to_ksdd': MVTEC_TO_KSDD,
            'mvtec_to_reinad': MVTEC_TO_ReinADtest, 'reinadtrain_to_reinadtest':ReinADtrain_TO_ReinADtest}



def main(args):

    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    

    if CLASSES['seen'][0] in MVTEC.CLASS_NAMES:  # from mvtec to other datasets
        train_dataset = MVTEC(args.train_dataset_dir, class_name=CLASSES['seen'], train=False, 
                               normalize="w50",
                               img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    elif CLASSES['seen'][0] in SELFTRAINDATA.CLASS_NAMES:  # from reinad to other datasets
        train_dataset = SELFTRAINDATA(args.train_dataset_dir, class_name=CLASSES['seen'], train=False, 
                               normalize="w50",
                               img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )


    backbone = timm.create_model('wide_resnet50_2', features_only=True, pretrained=True).eval()  
    backbone = backbone.to(args.device)
    feat_ids = list(range(3, 17))
    nbottlenecks = [3, 4, 6, 3] 
    bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
    lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
    stack_ids = torch.tensor(lids).bincount().__reversed__().cumsum(dim=0)[:3]
    hpn_learner = VATLearner(inch = [x for x in reversed(nbottlenecks[-3:])],shot=args.num_ref_shot)
    hpn_learner.to(args.device)
    optimizer = torch.optim.Adam(list(hpn_learner.parameters()), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 90], gamma=0.1)


    start_epoch = 0        
    best_pro = 0

    if args.resume and os.path.isfile(args.resume_path):
        print(f'>>> resume from {args.resume_path} …')
        ckpt = torch.load(args.resume_path, map_location=args.device)
        hpn_learner.load_state_dict(ckpt['hpn_learner'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1 
        best_pro = ckpt['best_pro']
        print(f'>>> Resume complete: starting from epoch {start_epoch}, historical best AUPRO = {best_pro:.4f}')
    else:
        print('>>> --resume not specified or file does not exist, starting training from scratch.')

    for epoch in range(start_epoch, args.epochs):
        hpn_learner.train()

        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for step, batch in enumerate(train_loader):

            loss = 0
            progress_bar.update(1)
            images, _, masks, class_names = batch
            images = images.to(args.device)
            masks = masks.to(args.device)
            
            with torch.no_grad():
                features = extract_feat_res(images, backbone, feat_ids, bottleneck_ids, lids)
            ref_features = get_mc_reference_features(backbone, args.train_dataset_dir, class_names, images.device, args.num_ref_shot, feat_ids, bottleneck_ids, lids)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features, args.num_ref_shot)

            if args.num_ref_shot == 1:
                corr = multilayer_correlation(features[-stack_ids[-1]:], mfeatures[-stack_ids[-1]:], stack_ids, args.num_ref_shot, is_test=False)
                pred = hpn_learner(corr, stack_feats(features, stack_ids), stack_feats(mfeatures, stack_ids), _, istrain=True)
                pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)

            else:
                all_support_list = []
                for feat in mfeatures:
                    B, n, C, H, W = feat.shape
                    y = feat.permute(0, 2, 3, 4, 1)
                    y = y.reshape(B, C, H, W * n)
                    all_support_list.append(y)

                all_pred_list = []
                for i in range(args.num_ref_shot):
                    ref_list_i = []
                    for t in mfeatures:
                        ref_list_i = [t[:, i, ...] for t in mfeatures]
                    corr = multilayer_correlation(features[-stack_ids[-1]:], ref_list_i[-stack_ids[-1]:], stack_ids, args.num_ref_shot, is_test=False)
                    pred = hpn_learner(corr, stack_feats(features, stack_ids), stack_feats(ref_list_i, stack_ids), stack_feats(all_support_list, stack_ids), istrain=False)
                    pred = F.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)
                    all_pred_list.append(pred)
                all_pred = torch.stack(all_pred_list, dim=0)
                pred = torch.mean(all_pred, dim=0)

            focal_loss = BinaryFocalLoss(alpha=1, gamma=2)
            loss = focal_loss(pred, masks.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
        scheduler.step()
        progress_bar.close()

        
        if (epoch + 1) % args.eval_freq == 0:
            
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
                    test_dataset = SELFDATA(args.test_dataset_dir, class_name=class_name, train=False,
                                           normalize='w50',
                                           img_size=512, crp_size=512, msk_size=512, msk_crp_size=512)                
                else:
                    raise ValueError('Unrecognized class name: {}'.format(class_name))
                test_loader = DataLoader(
                    test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=False
                )
                metrics = validate(args, backbone, hpn_learner, test_loader, test_ref_features[class_name], args.device, feat_ids, bottleneck_ids, lids, stack_ids, args.num_ref_shot)
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics['scores']
                
                print("Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                    epoch, class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
                s_res.append(metrics['scores'])
            
            s_res = np.array(s_res)
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(s_res, axis=0)
            print('(Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))

            if pix_aupro > best_pro:
            
                best_pro = pix_aupro
                state_dict = {
                    'epoch'      : epoch,             
                    'best_pro'   : best_pro,           
                    'hpn_learner': hpn_learner.state_dict(),
                    'optimizer'  : optimizer.state_dict(),
                    'scheduler'  : scheduler.state_dict()
                }
                os.makedirs(args.checkpoint_path, exist_ok=True)
                save_path = os.path.join(args.checkpoint_path, f'{args.setting}_{args.num_ref_shot}_checkpoints{epoch}.pth')
                torch.save(state_dict, save_path)
                print(f'>>> The latest model has been saved to {save_path}')

                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="mvtec_to_reinad")
    parser.add_argument('--train_dataset_dir', type=str, default="")
    parser.add_argument('--test_dataset_dir', type=str, default="")
    parser.add_argument('--test_ref_feature_dir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    parser.add_argument('--num_ref_shot', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='若指定则从 checkpoint 继续训练')
    parser.add_argument('--resume_path', type=str, default='', help='checkpoint 路径')


    
    args = parser.parse_args()
    init_seeds(42)
    
    main(args)