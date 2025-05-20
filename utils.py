import os
import random
from typing import List, Dict
from PIL import Image
import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import torch
from torch import Tensor
import torchvision.transforms as T
from datasets.mvtec import MVTEC
from datasets.visa import VISA
from ReinAD.datasets.ReinADtrain import SELFTRAINDATA


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.act1.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].act1.forward(feat)#### layer1
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].act2.forward(feat)#### layer2
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res
        feat = backbone.__getattr__('layer%d' % lid)[bid].act3.forward(feat)#### layer3
        if hid + 1 in feat_ids:
            feats.append(feat.clone())

    return feats
    
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     

def get_random_normal_images(root, class_name, num_shot=1):
    if class_name in MVTEC.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'train', 'good')
    elif class_name in VISA.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'Data', 'Images', 'Normal')
    elif class_name in SELFTRAINDATA.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'Data', 'Images', 'Normal')
    else:
        raise ValueError('Unrecognized class_name!')
    filenames = os.listdir(root_dir)
    n_idxs = np.random.randint(len(filenames), size=num_shot)
    n_idxs = n_idxs.tolist()
    normal_paths = []
    for n_idx in n_idxs:
        normal_paths.append(os.path.join(root_dir, filenames[n_idx]))
    
    return normal_paths

def get_mc_reference_features(backbone, root, class_names, device, num_shot, feat_ids, bottleneck_ids, lids):
    """
    Get reference features for multiple classes.
    """
    reference_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = extract_feat_res(images, backbone, feat_ids, bottleneck_ids, lids)
            reference_features[class_name] = features
    return reference_features


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(512, T.InterpolationMode.BICUBIC),
                T.CenterCrop(512),
                T.ToTensor(),
                T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)


def get_mc_matched_ref_features(features: List[Tensor], class_names: List[str],
                                ref_features: Dict[str, List[Tensor]], num_shot) -> List[Tensor]:
    """
    Get matched reference features for multiple classes.
    """
    matched_ref_features = [[] for _ in range(len(features))]
    diff_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names): 
        ref_features_c = ref_features[c] 
        for layer_id in range(len(features)): 
            coreset = ref_features_c[layer_id]
            index_feats = coreset.squeeze(0)
            matched_ref_features[layer_id].append(index_feats)
    if num_shot == 1 :        
        matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    else:
        matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


def calculate_metrics(scores, labels, gt_masks, pro=True, only_max_value=True, top_percent=0.2):
    """
    Args:
        scores (np.ndarray): shape (N, H, W).
        labels (np.ndarray): shape (N, ), 0 for normal, 1 for abnormal.
        gt_masks (np.ndarray): shape (N, H, W).
        only_max_value (bool): if True, use max score; else use top_percent mean.
        top_percent (float): percentage of top scores to average if only_max_value=False.
    """
    # pixel-level metrics
    pix_ap = round(average_precision_score(gt_masks.flatten(), scores.flatten()), 5)
    precisions, recalls, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    pix_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
    pix_auc = round(roc_auc_score(gt_masks.flatten(), scores.flatten()), 5)

    # image-level scoring
    n, h, w = scores.shape
    img_scores = []

    if only_max_value:
        img_scores = scores.reshape(n, -1).max(axis=1)
    else:
        k = max(1, int(h * w * top_percent))
        sorted_scores = np.sort(scores.reshape(n, -1), axis=1)  
        topk_scores = sorted_scores[:, -k:]  
        img_scores = topk_scores.mean(axis=1)

    # image-level metrics
    img_ap = round(average_precision_score(labels, img_scores), 5)
    precisions, recalls, _ = precision_recall_curve(labels, img_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
    img_auc = round(roc_auc_score(labels, img_scores), 5)

    # pixel-level aupro
    pix_aupro = calculate_aupro(gt_masks, scores) if pro else -1

    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro


def get_image_scores(scores, topk=1):
    scores_ = torch.from_numpy(scores)
    img_scores = torch.topk(scores_.reshape(scores_.shape[0], -1), topk, dim=1)[0]
    img_scores = torch.mean(img_scores, dim=1)
    img_scores = img_scores.cpu().numpy()
        
    return img_scores


def calculate_aupro(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs.shape[0] <= 2:
        return 0.5
    else:
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc

def multilayer_correlation(query_feats, support_feats, stack_ids, num_shot, is_test=True):
    eps = 1e-5
    corrs = []
    for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
    
        #if num_shot == 1:
        bsz, ch, hb, wb = support_feat.size()
        support_feat = support_feat.view(bsz, ch, -1)
        support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

        bsz, ch, ha, wa = query_feat.size()
        query_feat = query_feat.view(bsz, ch, -1)
        query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

        corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
        corr = corr.clamp(min=0)
        corrs.append(corr)

    corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
    corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
    corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

    return [corr_l4, corr_l3, corr_l2]

def stack_feats(feats, stack_ids):

    feats_l4 = torch.stack(feats[-stack_ids[0]:]).transpose(0, 1)
    feats_l3 = torch.stack(feats[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1)
    feats_l2 = torch.stack(feats[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1)
    feats_l1 = torch.stack(feats[:-stack_ids[2]]).transpose(0, 1)

    return [feats_l4, feats_l3, feats_l2, feats_l1]


def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=1):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
        layer4_refs = np.load(os.path.join(root_dir, class_name, 'layer4.npy'))
        layer5_refs = np.load(os.path.join(root_dir, class_name, 'layer5.npy'))
        layer6_refs = np.load(os.path.join(root_dir, class_name, 'layer6.npy'))
        layer7_refs = np.load(os.path.join(root_dir, class_name, 'layer7.npy'))
        layer8_refs = np.load(os.path.join(root_dir, class_name, 'layer8.npy'))
        layer9_refs = np.load(os.path.join(root_dir, class_name, 'layer9.npy'))
        layer10_refs = np.load(os.path.join(root_dir, class_name, 'layer10.npy'))
        layer11_refs = np.load(os.path.join(root_dir, class_name, 'layer11.npy'))
        layer12_refs = np.load(os.path.join(root_dir, class_name, 'layer12.npy'))
        layer13_refs = np.load(os.path.join(root_dir, class_name, 'layer13.npy'))
        layer14_refs = np.load(os.path.join(root_dir, class_name, 'layer14.npy'))

        layer1_refs = torch.from_numpy(layer1_refs).to(device)
        layer2_refs = torch.from_numpy(layer2_refs).to(device)
        layer3_refs = torch.from_numpy(layer3_refs).to(device)
        layer4_refs = torch.from_numpy(layer4_refs).to(device)
        layer5_refs = torch.from_numpy(layer5_refs).to(device)
        layer6_refs = torch.from_numpy(layer6_refs).to(device)
        layer7_refs = torch.from_numpy(layer7_refs).to(device)
        layer8_refs = torch.from_numpy(layer8_refs).to(device)
        layer9_refs = torch.from_numpy(layer9_refs).to(device)
        layer10_refs = torch.from_numpy(layer10_refs).to(device)
        layer11_refs = torch.from_numpy(layer11_refs).to(device)
        layer12_refs = torch.from_numpy(layer12_refs).to(device)
        layer13_refs = torch.from_numpy(layer13_refs).to(device)
        layer14_refs = torch.from_numpy(layer14_refs).to(device)
        
        refs[class_name] = (layer1_refs, layer2_refs, layer3_refs,layer4_refs, layer5_refs, layer6_refs,layer7_refs, layer8_refs, layer9_refs,layer10_refs, layer11_refs, layer12_refs, layer13_refs, layer14_refs)

    return refs

def get_near_features(features, ref_features):
    matched_ref_features = []
    B, C, H, W = features.shape
    
    for i in range(B):
        #print(ref_features[i].shape)
        feature1 = features[i].unsqueeze(0).permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature1, p=2, dim=1) 
        coreset1 = ref_features[i].unsqueeze(0).permute(0, 2, 3, 1).reshape(-1, C).contiguous() # (N2, C).
        #coreset1 = ref_features[i].permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        coreset_n = F.normalize(coreset1, p=2, dim=1)
        dist = feature_n @ coreset_n.T  # (N1, N2)
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset1[cidx]
        index_feats = index_feats.permute(1, 0).reshape(C, H, W)
        matched_ref_features.append(index_feats)

    matched_ref_features = torch.stack(matched_ref_features, dim=0)
    
    return matched_ref_features