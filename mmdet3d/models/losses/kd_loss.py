'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-08-14 10:56:10
Email: haimingzhang@link.cuhk.edu.cn
Description: The losses used for knowledge distillation.
'''

import torch
import torch.nn as nn
from ..builder import LOSSES


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape[:4]
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap


@LOSSES.register_module()
class CriterionCWD(nn.Module):
    """Adapted from 'Channel-wise Distillation for Semantic Segmentation'

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 norm_type='none',
                 divergence='mse',
                 temperature=1.0,
                 with_mask=False,
                 loss_weight=1.0):
    
        super(CriterionCWD, self).__init__()
       

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            reduction = 'none' if with_mask else 'sum'
            self.criterion = nn.KLDivLoss(reduction=reduction)
            self.temperature = temperature
        self.divergence = divergence
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T, mask=None):
        
        n,c,h,w = preds_S.shape[:4]
        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        if mask is not None:
            loss = loss * mask
            loss = loss.sum()
        
        #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        #import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return self.loss_weight * loss * (self.temperature**2)
