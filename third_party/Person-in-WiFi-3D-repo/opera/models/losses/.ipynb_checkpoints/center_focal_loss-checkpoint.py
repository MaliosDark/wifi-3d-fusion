# Copyright (c) Hikvision Research Institute. All rights reserved.
import mmcv
import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from ..builder import LOSSES


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def center_focal_loss(pred, gt, mask=None):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory.

    Args:
        pred (Tensor): The prediction with shape [bs, c, h, w].
        gt (Tensor): The learning target of the prediction in gaussian
            distribution, with shape [bs, c, h, w].
        mask (Tensor): The valid mask. Defaults to None.
    """
    pos_inds = gt.eq(1).float()
    if mask is None:
        neg_inds = gt.lt(1).float()
    else:
        neg_inds = gt.lt(1).float() * mask.eq(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * \
        neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


@LOSSES.register_module()
class CenterFocalLoss(nn.Module):
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
        super(CenterFocalLoss, self).__init__()
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
        loss_reg = self.loss_weight * center_focal_loss(
            pred,
            target,
            weight,
            mask=mask,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
