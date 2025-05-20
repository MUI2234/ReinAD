import warnings
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from utils import calculate_metrics, extract_feat_res, multilayer_correlation, stack_feats

warnings.filterwarnings('ignore')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def validate(args, backbone, hpn_learner, test_loader, ref_features, device, feat_ids, bottleneck_ids, lids, stack_ids, num_shot):
    backbone.eval()
    hpn_learner.eval()
    
    label_list, gt_mask_list = [], []
    pred_list = []
    for idx, batch in enumerate(test_loader):
        image, label, mask, class_name = batch    
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        image = image.to(device)


        with torch.no_grad():
            features = extract_feat_res(image, backbone, feat_ids, bottleneck_ids, lids)
            num_of_image = features[0].shape[0]
            if num_shot == 1:
                ref_list = []
                for t in ref_features:
                    batched = t.repeat(num_of_image, 1, 1, 1)
                    ref_list.append(batched)

                corr = multilayer_correlation(features[-stack_ids[-1]:], ref_list[-stack_ids[-1]:], stack_ids, num_shot, is_test=True)
                
                pred = hpn_learner(corr, stack_feats(features, stack_ids), stack_feats(ref_list, stack_ids), stack_feats(ref_list, stack_ids), istrain=False)
                pred = torch.nn.functional.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)
                pred_list.append(pred.squeeze(1).cpu().numpy())
            else:
                
                ref_list = []
                for feat in ref_features:
                    B, C, H, W = feat.shape
                    y = feat.permute(1, 2, 3, 0)
                    y = y.reshape(1, C, H, W * B)
                    ref_list.append(y)
                all_support_list = []
                for t in ref_list:
                    support = t.repeat(num_of_image, 1, 1, 1)
                    all_support_list.append(support)

                all_pred_list = [[] for _ in range(num_shot)]
                for i in range(num_shot):
                    ref_list_i = []
                    for t in ref_features:
                        batched = t[i:i+1].repeat(num_of_image, 1, 1, 1)
                        ref_list_i.append(batched)

                    corr = multilayer_correlation(features[-stack_ids[-1]:], ref_list_i[-stack_ids[-1]:], stack_ids, num_shot, is_test=True)
                    pred = hpn_learner(corr, stack_feats(features, stack_ids), stack_feats(ref_list_i, stack_ids), stack_feats(all_support_list, stack_ids), istrain=False)
                    pred = F.interpolate(
                        pred,
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=False
                    )
                    np_pred = pred.squeeze(1).cpu()
                    all_pred_list[i].append(np_pred)
                m = len(all_pred_list[0])
                for j in range(m):
                    tensors = [all_pred_list[i][j] for i in range(num_shot)] 
                    stacked = torch.stack(tensors, dim=0)
                    mean_tensor = stacked.mean(dim=0).numpy()
                    pred_list.append(mean_tensor)

            
    pred = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    gt_masks = np.concatenate(gt_mask_list, axis=0)


    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(pred, labels, gt_masks, pro=True, only_max_value=False)
    
    metrics = {}
    metrics['scores'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
    
    return metrics
