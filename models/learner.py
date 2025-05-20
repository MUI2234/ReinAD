import torch.nn as nn
import torch.nn.functional as F

from models.ours import OurModel


class VATLearner(nn.Module):
    def __init__(self, inch, shot):
        super(VATLearner, self).__init__()

        self.ours = OurModel(inch, shot)

    def forward(self, hypercorr_pyramid, query_feats, support_feats, all_support_feats, istrain=True):
        return self.ours(hypercorr_pyramid, query_feats, support_feats, all_support_feats, istrain)