# Copyright (c) Hikvision Research Institute. All rights reserved.
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weighted_loss

from ..builder import LOSSES


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def limb_loss(pred, gt, mask=None):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory.

    Args:
        pred (Tensor): The prediction with shape [bs, c, h, w].
        gt (Tensor): The learning target of the prediction in gaussian
            distribution, with shape [bs, c, h, w].
        mask (Tensor): The valid mask. Defaults to None.
    """
    connected_limbs = torch.tensor([[0,1], [0,2], [0,3], [2, 4], [2,5], [3,6],
                               [3,7], [4,8], [5,7], [5,9], [6,10], [7,11],
                               [9,12], [11,13]])
    pred = pred.reshape(-1,14,3)
    gt = gt.reshape(-1,14,3)
 
    vec_pred = pred[:,connected_limbs.transpose(0,1)[0],:] - pred[:,connected_limbs.transpose(0,1)[1],:]
    vec_gt = gt[:,connected_limbs.transpose(0,1)[0],:] - gt[:,connected_limbs.transpose(0,1)[1],:]
    limb_vector_loss =  (1 - F.cosine_similarity(vec_pred, vec_gt, dim=-1))
    limb_vector_loss = limb_vector_loss.reshape(-1,14)
    
    return limb_vector_loss


@LOSSES.register_module()
class LimbLoss(nn.Module):
    """CenterFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 reduction='none',
                 loss_weight=1.0):
        super(LimbLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction in gaussian
                distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            mask (Tensor): The valid mask. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_reg = self.loss_weight * limb_loss(
            pred,
            target,
            weight,
            mask=mask,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
