import torch
import torch.nn as nn
import torch.nn.functional as F

from models.swin import SwinTransformer2d, TransformerWarper2d
from models.our_conv4d import Interpolate4d, Encoder4D
from models.swin4d import SwinTransformer, TransformerWarper
from utils import get_near_features

import matplotlib.pyplot as plt

class UpsampleConv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, channels=(32, 192, 256)):  
        super().__init__()
        self.l3_to_l2 = UpsampleConv(channels[2], channels[1], scale_factor=2)  # 32 → 192
        self.l2_to_l1 = UpsampleConv(channels[1], channels[0], scale_factor=2)  # 192 → 256

    def forward(self, layer1_x, layer2_x, layer3_x):
        layer3_x_to_2 = self.l3_to_l2(layer3_x)  # [B, 32, 128, 128] → [B, 192, 64, 64]
        layer2_x_to_1 = self.l2_to_l1(layer2_x + layer3_x_to_2)  # [B, 192, 64, 64] → [B, 256, 32, 32]

        out1 = layer1_x + layer2_x_to_1 
        return out1
    
class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.mask_mlp = nn.Sequential(
            nn.Linear(32, 16),  
            nn.ReLU(inplace=True),
            nn.Linear(16, 1), 
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid() 
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).view(B,H*W,C)
        x = self.mask_mlp(x)  
        
        x = x.permute(0, 2, 1).view(B, 1, H, W)  #[B, C, H, W]

        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        return x


class OurModel(nn.Module):
    def __init__(self,
        inch=(3, 23, 4),
        num_shot=1,
    ):
        super().__init__()
        self.shot = num_shot
        self.encoders = nn.ModuleList([
            Encoder4D( # Encoder for conv_5
                corr_levels=(inch[0], 64, 128),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 1, 1),
                    (1, 1, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                group=(4, 8),
                residual=False
            ),
            Encoder4D( # Encoder for conv_4
                corr_levels=(inch[1], 64, 128),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                    (1, 1, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                group=(4, 8),
                residual=False
            ),
            Encoder4D( # Encoder for conv_3
                corr_levels=(inch[2], 32, 64, 128),
                kernel_size=(
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                    (1, 1, 2, 2),
                    (1, 1, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                    (1, 1, 1, 1),
                ),
                group=(2, 4, 8,),
                residual=False
            ),
        ])

        self.transformer = nn.ModuleList([
            TransformerWarper(SwinTransformer(
                corr_size=(8, 8, 8, 8),
                embed_dim=128,
                depth=4,
                num_head=1,
                window_size=4,
            )),
            TransformerWarper(SwinTransformer(
                corr_size=(16, 16, 8, 8),
                embed_dim=128,
                depth=2,
                num_head=1,
                window_size=4,
            )),
            TransformerWarper(SwinTransformer(
                corr_size=(32, 32, 8, 8),
                embed_dim=128,
                depth=2,
                num_head=1,
                window_size=4,
            )),
        ])

        self.upscale = nn.ModuleList([
            Interpolate4d(size=(16, 16), dim='query'),
            Interpolate4d(size=(32, 32), dim='query'),
            Interpolate4d(size=(64, 64), dim='query'),
        ])

        self.swin_decoder = nn.ModuleList([
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(32, 32), embed_dim=128 + 256, window_size=8, num_heads=[1])
                ),
                nn.Conv2d(128 + 256, 128, 1)
            ),
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(64, 64), embed_dim=128 + 128, window_size=8, num_heads=[1])
                ),
                nn.Conv2d(128 + 128, 128, 1)
            ),
            nn.Sequential(
                TransformerWarper2d(
                    SwinTransformer2d(img_size=(128, 128), embed_dim=128 + 64, window_size=8, num_heads=[1])
                ),
            ),         
        ])
        # 3*3 
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 1, (3, 3), padding=(1, 1), bias=True),
            nn.Sigmoid()
            ),  
            nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 1, (3, 3), padding=(1, 1), bias=True),
            nn.Sigmoid()
            ),     
            nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 1, (3, 3), padding=(1, 1), bias=True),
            nn.Sigmoid()
            ),
            nn.Sequential(
            nn.Conv2d(192, 64, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 1, (3, 3), padding=(1, 1), bias=True),
            nn.Sigmoid()
            ),       
        ])
        
        self.dropout2d = nn.Dropout2d(p=0.5)

        self.proj_query_feat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, 1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, 1),
                nn.ReLU(),
            ),            
            
        ]) 
        #self.fusion_module = MultiScaleFusion(channels=(32, 192, 256))
        #self.decoder = MaskDecoder()

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode='nearest') for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [F.interpolate(x, size=size, mode='nearest') for x, size in zip(feats, sizes)]
        return recoverd_feats
    
    def extract_last(self, x):
        return [k[:, -1] for k in x]

    def forward(self, hypercorr_pyramid, query_feats, support_feats, all_support_feats, istrain):
        
        _, query_feat4, query_feat3, query_feat2 = self.extract_last(query_feats)
        _, support_feat4, support_feat3, support_feat2 = self.extract_last(support_feats)#x_reshaped = x.view(B, l, C, H, W * n)
        if self.shot == 1:
            support_feat4 = get_near_features(query_feat4, support_feat4)
            support_feat3 = get_near_features(query_feat3, support_feat3)
            support_feat2 = get_near_features(query_feat2, support_feat2)
        else:
            _, all_support_feat4, all_support_feat3, all_support_feat2 = self.extract_last(all_support_feats)
            support_feat4 = get_near_features(query_feat4, all_support_feat4)
            support_feat3 = get_near_features(query_feat3, all_support_feat3)
            support_feat2 = get_near_features(query_feat2, all_support_feat2)

        query_feat4, query_feat3, query_feat2 = [
            self.proj_query_feat[i](x) for i, x in enumerate((query_feat4, query_feat3, query_feat2)) 
        ]
        support_feat4, support_feat3, support_feat2 = [
            self.proj_query_feat[i](x) for i, x in enumerate((support_feat4, support_feat3, support_feat2)) 
        ]
        query_feat4, query_feat3 = self.apply_dropout(self.dropout2d, query_feat4, query_feat3)
        support_feat4, support_feat3 = self.apply_dropout(self.dropout2d, support_feat4, support_feat3)

        
        corr5 = self.encoders[0](hypercorr_pyramid[0])[0]
        corr4 = self.encoders[1](hypercorr_pyramid[1])[0]
        corr3 = self.encoders[2](hypercorr_pyramid[2])[0]
        
        corr5 = corr5 + self.transformer[0](corr5)
        corr5_upsampled = self.upscale[0](corr5)

        corr4 = corr4 + corr5_upsampled
        corr4 = corr4 + self.transformer[1](corr4)
        corr4_upsampled = self.upscale[1](corr4)

        corr3 = corr3 + corr4_upsampled
        corr3 = corr3 + self.transformer[2](corr3)
        x_concatenated = corr3.mean(dim=(-2, -1))

        out1 = F.interpolate(self.decoder[0](x_concatenated), size=(128, 128), mode='bilinear', align_corners=True)

        combine_feat4 = support_feat4 - query_feat4
        combinex4 = torch.cat((x_concatenated, combine_feat4), dim=1)
        x_concatenated = self.swin_decoder[0](combinex4)
        out2 = F.interpolate(self.decoder[1](x_concatenated), size=(128, 128), mode='bilinear', align_corners=True)

        combine_feat3 = support_feat3 - query_feat3
        x_concatenated = F.interpolate(x_concatenated, size=(64, 64), mode='bilinear', align_corners=True)
        combinex3 = torch.cat((x_concatenated, combine_feat3), dim=1)
        x_concatenated = self.swin_decoder[1](combinex3)
        out3 = F.interpolate(self.decoder[2](x_concatenated), size=(128, 128), mode='bilinear', align_corners=True)

        combine_feat2 = support_feat2 - query_feat2
        x_concatenated = F.interpolate(x_concatenated, size=(128, 128), mode='bilinear', align_corners=True)
        combinex2 = torch.cat((x_concatenated, combine_feat2), dim=1)
        x_concatenated = self.swin_decoder[2](combinex2)
        out4 = self.decoder[3](x_concatenated)

        out = (out1 + out2 + out3 + out4) / 4      
        return out
    
