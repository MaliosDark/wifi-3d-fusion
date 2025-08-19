# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from opera.core.bbox.builder import BBOX_ASSIGNERS
from opera.core.bbox.match_costs import build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class PoseHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression oks cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt.
    - positive integer: positive sample, index (1-based) of assigned gt.

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        kpt_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        oks_weight (int | float, optional): The scale factor for regression
            oks cost. Default 1.0.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.0),
                 kpt_cost=dict(type='KptL1Cost', weight=1.0),
                 oks_cost=dict(type='OksCost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.kpt_cost = build_match_cost(kpt_cost)
        self.oks_cost = build_match_cost(oks_cost)


    def assign(self,
               cls_pred,
               kpt_pred,
               gt_labels,
               gt_keypoints,
               gt_areas,
               img_meta,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            kpt_pred (Tensor): Predicted keypoints with normalized coordinates
                (x_{i}, y_{i}), which are all in range [0, 1]. Shape
                [num_query, K*2].
            gt_labels (Tensor): Label of `gt_keypoints`, shape (num_gt,).
            gt_keypoints (Tensor): Ground truth keypoints with unnormalized
                coordinates [p^{1}_x, p^{1}_y, p^{1}_v, ..., \
                    p^{K}_x, p^{K}_y, p^{K}_v]. Shape [num_gt, K*3].
            gt_areas (Tensor): Ground truth mask areas, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_kpts = gt_keypoints.size(0), kpt_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = kpt_pred.new_full((num_kpts, ),
                                             -1,
                                             dtype=torch.long)
        assigned_labels = kpt_pred.new_full((num_kpts, ),
                                            -1,
                                            dtype=torch.long)
        if num_gts == 0 or num_kpts == 0:
            # No ground truth or keypoints, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
                
        # 2. compute the weighted costs
        # classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # keypoint regression L1 cost
        gt_keypoints_reshape = gt_keypoints.reshape(gt_keypoints.shape[0], -1,
                                                    3)
        valid_kpt_flag = gt_keypoints_reshape.new_ones(([num_gts, gt_keypoints_reshape.shape[1]]))
        kpt_pred_tmp = kpt_pred.clone().detach().reshape(
            kpt_pred.shape[0], -1, 3)

        kpt_cost = self.kpt_cost(kpt_pred_tmp, gt_keypoints_reshape,
                                 valid_kpt_flag)
        

        # keypoint OKS cost
        # kpt_pred_tmp = kpt_pred.clone().detach().reshape(
        #     kpt_pred.shape[0], -1, 2)

        # oks_cost = self.oks_cost(kpt_pred_tmp, gt_keypoints_reshape[..., :2],
        #                          valid_kpt_flag, gt_areas)
        # weighted sum of above three costs
        cost = cls_cost + kpt_cost 

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            kpt_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            kpt_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
