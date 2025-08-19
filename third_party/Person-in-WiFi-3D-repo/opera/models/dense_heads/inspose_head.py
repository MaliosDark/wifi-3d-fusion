# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d
from mmdet.core import multi_apply
from mmdet.core.post_processing import multiclass_nms
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from ..builder import HEADS, build_loss
from opera.core.keypoint import (distance2keypoint, draw_umich_gaussian,
                                 draw_short_range_offset, gaussian_radius,
                                 transpose_and_gather_feat)

INF = 1e8


@HEADS.register_module
class InsPoseHead(AnchorFreeHead):
    """Head of InsPose: Instance-Aware Networks for Single-Stage Multi-Person
    Pose Estimation.

    More details can be found in the `paper
    <https://arxiv.org/abs/2107.08982>`_ .

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of stacking convs of the head.
        num_keypoints (int): Number of joints of each pose.
        strides (tuple): Downsample factor of each feature map.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 feat_channels_kpt=512,
                 stacked_convs_kpt=4,
                 feat_channels_hm=512,
                 stacked_convs_hm=3,
                 num_keypoints=17,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 centerness_on_reg=True,
                 regression_normalize=True,
                 with_hm_loss=True,
                 min_overlap_hm=0.7,
                 min_hm_radius=0,
                 max_hm_radius=3,
                 min_overlap_kp=0.9,
                 min_offset_radius=0,
                 max_offset_radius=3,
                 ae_loss_type='exp',
                 ae_loss_weight=1.0,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.75,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_hm=dict(type='CenterFocalLoss', loss_weight=1.0),
                 loss_weight_offset=1.0,
                 unvisible_weight=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=dict(
                     nms_pre=1000,
                     score_thr=0.05,
                     nms=dict(type='soft_nms', iou_threshold=0.3),
                     mask_thresh=0.5,
                     max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.feat_channels_kpt = feat_channels_kpt
        self.stacked_convs_kpt = stacked_convs_kpt
        self.feat_channels_hm = feat_channels_hm
        self.stacked_convs_hm = stacked_convs_hm
        self.shared_channels = 128
        self.num_keypoints = num_keypoints
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_hm = build_loss(loss_hm)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.centerness_on_reg = centerness_on_reg
        self.regression_normalize = regression_normalize
        self.loss_weight_offset = loss_weight_offset
        self.unvisible_weight = unvisible_weight
        self.with_hm_loss = with_hm_loss
        self.min_overlap_hm = min_overlap_hm
        self.min_hm_radius = min_hm_radius
        self.max_hm_radius = max_hm_radius
        self.min_overlap_kp = min_overlap_kp
        self.min_offset_radius = min_offset_radius
        self.max_offset_radius = max_offset_radius
        self.ae_loss_type = ae_loss_type
        self.ae_loss_weight = ae_loss_weight
        self.dcn_kernel = 3
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float32)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x],
                                   axis=1).reshape(-1)
        self.dcn_base_offset = torch.Tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.gradient_mul = 0.1
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
        self.sigmas = torch.Tensor(sigmas)
        if self.with_hm_loss:
            self.hm_feat = None
            self.ae_feat = None
            self.hm_offset_feat = None

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.ctrl_convs = nn.ModuleList()
        self.shared_convs = nn.ModuleList()
        self.kpt_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.ctrl_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        for i in range(4):
            chn = self.in_channels if i == 0 else self.shared_channels
            self.shared_convs.append(
                ConvModule(
                    chn,
                    self.shared_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.shared_convs.append(
            ConvModule(
                self.shared_channels,
                8,
                3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=dict(type='GN', num_groups=2, requires_grad=True),
                bias=self.norm_cfg is None))
        for i in range(self.stacked_convs_kpt):
            chn = self.in_channels if i == 0 else self.feat_channels_kpt
            self.kpt_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels_kpt,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.conv_cls = nn.Conv2d(
            self.feat_channels * 2, self.cls_out_channels, 3, padding=1)
        # conv1(10x1x1x8+8)-conv2(8x1x1x8+8)-conv3(8x1x1x17+17)
        self.controller = nn.Conv2d(self.feat_channels * 2, 313, 3, padding=1)
        self.kpt_offset = nn.Conv2d(
            self.feat_channels_kpt, 2 * self.num_keypoints, 3, padding=1)
        self.cls_star_conv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            padding=self.dcn_pad)
        self.cls_star_gn = nn.GroupNorm(32, self.feat_channels)
        self.ctr_star_conv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            padding=self.dcn_pad)
        self.ctr_star_gn = nn.GroupNorm(32, self.feat_channels)
        self.relu = nn.ReLU(inplace=True)

        if self.regression_normalize:
            self.scales = [None for _ in self.strides]
        else:
            self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        if self.with_hm_loss:
            self.hm_convs = nn.ModuleList()
            for i in range(self.stacked_convs_hm):
                chn = self.in_channels if i == 0 else self.feat_channels_hm
                self.hm_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels_hm,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None))
            self.hm_pred = nn.Conv2d(
                self.feat_channels_hm, self.num_keypoints, 3, padding=1)
            self.ae_pred = nn.Conv2d(
                self.feat_channels_hm, self.num_keypoints, 3, padding=1)
            # keypoint-aware short-range centripetal offset
            self.hm_offset = nn.Conv2d(
                self.feat_channels_hm, 2 * self.num_keypoints, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.ctrl_convs:
            normal_init(m.conv, std=0.01)
        for m in self.shared_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kpt_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.controller, std=0.01)
        normal_init(self.kpt_offset, std=0.01)
        normal_init(self.cls_star_conv, std=0.1)
        normal_init(self.ctr_star_conv, std=0.1)
        if self.with_hm_loss:
            for m in self.hm_convs:
                normal_init(m.conv, std=0.01)
            bias_hm = bias_init_with_prob(0.1)
            normal_init(self.hm_pred, std=0.01, bias=bias_hm)
            normal_init(self.ae_pred, std=0.01)
            normal_init(self.hm_offset, std=0.01)

    def forward(self, feats):
        shared_feat = feats[0]
        for shared_layer in self.shared_convs:
            shared_feat = shared_layer(shared_feat)
        if self.with_hm_loss:
            hm_shared_feat = feats[0]
            for hm_layer in self.hm_convs[:-1]:
                hm_shared_feat = hm_layer(hm_shared_feat)
            hm_shared_feat = F.interpolate(hm_shared_feat, scale_factor=2,
                mode='bilinear', align_corners=False)
            hm_shared_feat = self.hm_convs[-1](hm_shared_feat)
            self.hm_feat = self.hm_pred(hm_shared_feat)
            self.ae_feat = self.ae_pred(hm_shared_feat)
            self.hm_offset_feat = self.hm_offset(hm_shared_feat)
        return multi_apply(self.forward_single, feats, self.scales) + \
                            (shared_feat, )

    def forward_single(self, x, scale):
        cls_feat = x
        ctrl_feat = x
        kpt_feat = x

        for kpt_layer in self.kpt_convs:
            kpt_feat = kpt_layer(kpt_feat)
        offset_pred = self.kpt_offset(kpt_feat).float()

        dcn_offset = self.star_dcn_offset(offset_pred, self.gradient_mul)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_feat_star = self.cls_star_conv(cls_feat, dcn_offset)
        cls_feat_star = self.relu(self.cls_star_gn(cls_feat_star))
        cls_score = self.conv_cls(torch.cat([cls_feat, cls_feat_star], dim=1))

        for ctrl_layer in self.ctrl_convs:
            ctrl_feat = ctrl_layer(ctrl_feat)
        ctrl_feat_star = self.ctr_star_conv(ctrl_feat, dcn_offset)
        ctrl_feat_star = self.relu(self.ctr_star_gn(ctrl_feat_star))
        controller = self.controller(
            torch.cat([ctrl_feat, ctrl_feat_star], dim=1))

        return cls_score, controller, offset_pred

    def star_dcn_offset(self, offset_pred, gradient_mul):
        """Compute the star deformable conv offsets.

        Args:
            offset_pred (Tensor): Predicted keypoint offset (delta_x, delta_y) * 17.
            gradient_mul (float): Gradient multiplier.

        Returns:
            dcn_offset (Tensor): The offsets for deformable convolution.
        """
        N, C, H, W = offset_pred.size()

        dcn_base_offset = self.dcn_base_offset.type_as(offset_pred)
        offset_pred_grad_mul = (1 - gradient_mul) * offset_pred.detach() + \
            gradient_mul * offset_pred
        offset_pred_grad_mul = offset_pred_grad_mul.view(N, C // 2, 2, H, W)
        offset_pred_grad_mul = offset_pred_grad_mul[:, :, [1, 0], :, :]
        # 0-nose, 5-left_shoulder, 6-right_shoulder, 9-left_wrist, 10-right_wrist
        # 11-left_hip, 12-right_hip, 15-left_ankle, 16-right_ankle
        offset_pred_select = offset_pred_grad_mul[
            :, [0, 5, 6, 9, 10, 11, 12, 15, 16], :, :].view(N, -1, H, W)
        dcn_offset = offset_pred_select - dcn_base_offset
        return dcn_offset

    @force_fp32(apply_to=('cls_scores'))
    def loss(self,
             cls_scores,
             controllers,
             offset_preds,
             shared_feat,
             gt_bboxes,
             gt_labels,
             gt_masks,
             gt_keypoints,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(controllers) == len(offset_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes,
            dtype=cls_scores[0].dtype, device=cls_scores[0].device)
        labels, bbox_targets, ins_inds, img_inds = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, gt_keypoints)
        if self.regression_normalize:
            bbox_targets = [
                bbox_target / stride
                for bbox_target, stride in zip(bbox_targets, self.strides)
            ]
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_controller = [
            controller.permute(0, 2, 3, 1).reshape(-1, 313)
            for controller in controllers
        ]
        flatten_offset_preds = [
            offset_pred.permute(0, 2, 3, 1).reshape(-1, 2 * self.num_keypoints)
            for offset_pred in offset_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_offset_preds = torch.cat(flatten_offset_preds)
        flatten_controller = torch.cat(flatten_controller)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_ins_inds = torch.cat(ins_inds)
        flatten_img_inds = torch.cat(img_inds)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_strides = torch.cat(
            [points.new_tensor(stride)[None].repeat(num_imgs * points.shape[0])
            for stride, points in zip(self.strides, all_level_points)]
        )
        flatten_coord_normalize = torch.cat(
            [points.new_tensor(2 ** i * 64).repeat(num_imgs * points.shape[0])
            for i, points in enumerate(all_level_points)]
        )
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        oks_weight = torch.ones_like(
            flatten_cls_scores) * self.unvisible_weight
        iou_weight = torch.ones_like(flatten_cls_scores)
        pos_controller = flatten_controller[pos_inds]
        pos_points = flatten_points[pos_inds]
        pos_coord_normalize = flatten_coord_normalize[pos_inds]
        # keypoint loss
        pos_oks_weight = oks_weight[pos_inds]
        pos_offset_preds = flatten_offset_preds[pos_inds]
        # get keypoint target
        pos_strides = flatten_strides[pos_inds]
        pos_ins_inds = flatten_ins_inds[pos_inds]
        pos_img_inds = flatten_img_inds[pos_inds]
        loss_keypoint = 0
        avg_factor = 0
        loss_keypoint_reg = 0
        avg_factor_reg = 0
        for i in range(num_imgs):
            cur_ind = pos_img_inds == i
            cur_pos_oks_weight = pos_oks_weight[cur_ind]
            cur_pos_controller = pos_controller[cur_ind]
            cur_pos_offset = pos_offset_preds[cur_ind]
            cur_pos_ins_inds = pos_ins_inds[cur_ind]
            cur_pos_point = pos_points[cur_ind]
            cur_pos_coord_normalize = pos_coord_normalize[cur_ind]
            cur_pos_stride = pos_strides[cur_ind]
            # loss_keypoint of conditional convolution prediction
            if cur_pos_controller.size(0) > 0:
                heatmap_preds = []
                for j in range(cur_pos_controller.size(0)):
                    rel_coord_map = self.get_coord_map(
                        shared_feat, cur_pos_point[j],
                        cur_pos_coord_normalize[j])
                    heatmap_preds.append(
                        self.kpt_fcn_head(cur_pos_controller[j],
                                        shared_feat[i][None], rel_coord_map))
                heatmap_preds = torch.cat(heatmap_preds)
                heatmap_preds_reshape = heatmap_preds.reshape(-1,
                    shared_feat.size(2) * shared_feat.size(3))
                # get heatmap targets
                gt_keypoint = gt_keypoints[i].clone()
                gt_mask = heatmap_preds.new_tensor(gt_masks[i].to_ndarray())
                gt_mask_area = gt_mask.sum(-1).sum(-1)
                gt_keypoint = gt_keypoint.reshape(gt_keypoint.shape[0], -1, 3)
                gt_keypoint[..., :2] = torch.floor(gt_keypoint[..., :2] / 8)
                # x_coord
                assert gt_keypoint[..., 0].max() <= shared_feat.size(3)
                # y_coord
                assert gt_keypoint[..., 1].max() <= shared_feat.size(2)
                gt_keypoint = gt_keypoint[cur_pos_ins_inds].reshape(-1, 3)
                gt_mask_area = gt_mask_area[cur_pos_ins_inds]
                gt_bbox = gt_bboxes[i][cur_pos_ins_inds]
                pos_valid = gt_keypoint[:, 2] > 0
                avg_factor += pos_valid.sum().float()
                keypoint_position_targets = gt_keypoint[:, 1] * \
                                            shared_feat.size(3) + \
                                            gt_keypoint[:, 0]
                keypoint_position_targets = keypoint_position_targets.long()
                # softmax cross entropy loss
                if pos_valid.sum() > 0:
                    loss_keypoint += F.cross_entropy(
                        heatmap_preds_reshape[pos_valid],
                        keypoint_position_targets[pos_valid],
                        reduction='sum')
                else:
                    loss_keypoint += heatmap_preds_reshape[pos_valid].sum()

                # pose score loss (BEC Loss)
                pos = heatmap_preds.reshape(
                    heatmap_preds.size(0), heatmap_preds.size(1),
                    -1).argmax(dim=2)
                x_pos = pos % heatmap_preds.size(3)
                y_pos = (pos - x_pos) // heatmap_preds.size(3)
                pred_keypoint = torch.cat(
                    [x_pos[:, :, None], y_pos[:, :, None]], dim=-1).float()
                # TODO: filter gts that has no valid keypoint
                gt_keypoint_reshape = gt_keypoint.reshape(-1, 17, 3)
                valid_idx = gt_keypoint_reshape[..., 2].sum(1) > 0
                if valid_idx.sum() > 0:
                    oks_score = self.compute_oks(
                        pred_keypoint.clone().detach()[valid_idx],
                        gt_keypoint_reshape[valid_idx],
                        gt_mask_area[valid_idx],
                        gt_bbox[valid_idx])
                    cur_pos_oks_weight[valid_idx] = oks_score[:, None]
            else:
                loss_keypoint += cur_pos_controller.sum() * shared_feat[i].sum()
            pos_oks_weight[cur_ind] = cur_pos_oks_weight

            # loss_keypoint of regressing directly
            if cur_pos_offset.size(0) > 0:
                pos_keypoints = gt_keypoints[i][cur_pos_ins_inds]
                pos_keypoints = pos_keypoints.reshape(pos_keypoints.shape[0],
                                                      -1, 3)
                offset_targets = self.keypoint_target(pos_keypoints[..., :2],
                                                      cur_pos_point,
                                                      cur_pos_stride)
                pos_valid = (pos_keypoints[..., 2] > 0).view(-1)
                # count every x and y
                avg_factor_reg += pos_valid.sum().float() * 2
                offset_preds = cur_pos_offset.reshape(-1, 2)
                offset_targets = offset_targets.reshape(-1, 2)
                loss_keypoint_reg += F.l1_loss(
                    offset_preds[pos_valid],
                    offset_targets[pos_valid],
                    reduction='sum')
            else:
                loss_keypoint_reg += cur_pos_offset.sum()
        oks_weight[pos_inds] = pos_oks_weight
        cls_weight = oks_weight
        pos_labels = (flatten_labels >= 0) & (flatten_labels < bg_class_ind)
        loss_cls = self.loss_cls(flatten_cls_scores,
            cls_weight * pos_labels[:, None].float(),
            avg_factor=num_pos + num_imgs)
        loss_keypoint = loss_keypoint / avg_factor \
            if avg_factor > 0 else loss_keypoint
        loss_keypoint_reg = loss_keypoint_reg / avg_factor_reg \
            if avg_factor_reg > 0 else loss_keypoint_reg
        loss_keypoint_reg *= self.loss_weight_offset

        if self.with_hm_loss:
            loss_hm, loss_ae_pull, loss_ae_push, loss_hm_offset = \
                self.heatmap_ae_offset_loss(
                    self.hm_feat,
                    self.ae_feat,
                    self.hm_offset_feat,
                    gt_keypoints,
                    gt_labels,
                    gt_bboxes,
                    img_metas)
            return dict(
                loss_cls=loss_cls,
                loss_keypoint=loss_keypoint,
                loss_keypoint_reg=loss_keypoint_reg,
                loss_heatmap=loss_hm,
                loss_ae_pull=loss_ae_pull,
                loss_ae_push=loss_ae_push,
                loss_heatmap_offset=loss_hm_offset)

        return dict(
            loss_cls=loss_cls,
            loss_keypoint=loss_keypoint,
            loss_keypoint_reg=loss_keypoint_reg)

    def compute_oks(self, pred_keypoints, gt_keypoints, gt_areas, gt_bboxes):
        sigmas = self.sigmas.type_as(gt_keypoints)
        vars = (sigmas * 2) ** 2
        xd, yd = pred_keypoints[..., 0], pred_keypoints[..., 1]
        valid = gt_keypoints[..., 2] > 0
        # any one keypoint is visible
        dx = xd - gt_keypoints[..., 0]
        dy = yd - gt_keypoints[..., 1]
        # find all keypoint is not visible and reset dx, dy, valid
        assert (valid.sum(1) > 0).all()
        # calculate (1/2)*(x^2+y^2)/sigma^2/area
        e = (dx ** 2 + dy ** 2) / vars / (gt_areas[:, None] + 1e-4) / 2
        sim = torch.exp(-e)
        sim[~valid] = 0
        oks = sim.sum(1) / valid.sum(1).float()
        return oks

    @force_fp32(apply_to=('cls_scores'))
    def get_bboxes(self,
                   cls_scores,
                   controllers,
                   offset_preds,
                   shared_feats,
                   imgs,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(offset_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, cls_scores[0].dtype,
                                      cls_scores[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            controller_pred_list = [
                controllers[i][img_id].detach() for i in range(num_levels)
            ]
            offset_pred_list = [
                offset_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img = imgs[img_id]
            img_meta = img_metas[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            shared_feat = shared_feats[img_id]
            proposals = self._get_bboxes_single(cls_score_list,
                                                controller_pred_list,
                                                offset_pred_list, shared_feat,
                                                mlvl_points, img, img_meta,
                                                img_shape, scale_factor, cfg,
                                                rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           controllers,
                           offset_preds,
                           shared_feat,
                           points,
                           img,
                           img_meta,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        assert len(cls_scores) == len(offset_preds) == len(points)
        mlvl_scores = []
        mlvl_controller = []
        mlvl_keypoints = []
        mlvl_points = []
        mlvl_coord_normalizes = []
        for cls_score, controller, offset_pred, point, stride in zip(
                cls_scores, controllers, offset_preds, points, self.strides):
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            controller = controller.permute(1, 2, 0).reshape(-1, 313)
            offset_pred = offset_pred.permute(1, 2, 0).reshape(
                -1, 2 * self.num_keypoints)
            coord_normalize = point.new_ones(point.size(0), 1) * stride * 8
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                topk_scores, topk_inds = max_scores.topk(nms_pre)
                point = point[topk_inds, :]
                scores = scores[topk_inds, :]
                controller = controller[topk_inds]
                offset_pred = offset_pred[topk_inds, :]
                coord_normalize = coord_normalize[topk_inds, :]
            # normalized by stride
            offset_pred = offset_pred * stride
            keypoints = distance2keypoint(
                point, offset_pred, max_shape=img_shape)
            mlvl_scores.append(scores)
            mlvl_controller.append(controller)
            mlvl_keypoints.append(keypoints)
            mlvl_points.append(point)
            mlvl_coord_normalizes.append(coord_normalize)
        mlvl_keypoints = torch.cat(mlvl_keypoints)
        if rescale:
            mlvl_keypoints /= mlvl_keypoints.new_tensor(scale_factor[:2])
        else:
            # keypoint heatmap visualization
            pass
            # self.show_keypoint_heatmap(self.hm_feat.sigmoid(), 4, img, img_meta)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_controller = torch.cat(mlvl_controller)
        mlvl_points = torch.cat(mlvl_points)
        mlvl_coord_normalizes = torch.cat(mlvl_coord_normalizes)
        padding = mlvl_scores.new_ones(mlvl_keypoints.shape[0],
                                       mlvl_keypoints.shape[1], 1)
        mlvl_keypoints = torch.cat([mlvl_keypoints, padding], dim=2)
        mlvl_bboxes = self.get_pseudo_bbox(mlvl_keypoints)
        det_bboxes, det_labels, nms_inds = multiclass_nms(mlvl_bboxes,
                                                          mlvl_scores,
                                                          cfg.score_thr,
                                                          cfg.nms,
                                                          cfg.max_per_img,
                                                          return_inds=True)
        det_controllers = mlvl_controller[nms_inds]
        det_keypoints = mlvl_keypoints[nms_inds]
        det_points = mlvl_points[nms_inds]
        det_coord_normalizes = mlvl_coord_normalizes[nms_inds]

        if det_bboxes.size(0) > 0:
            kpt_heatmaps = []
            for i in range(det_controllers.size(0)):
                rel_coord_map = self.get_coord_map(shared_feat[None],
                                                   det_points[i],
                                                   det_coord_normalizes[i])
                kpt_heatmap = self.kpt_fcn_head(det_controllers[i],
                                                shared_feat[None],
                                                rel_coord_map)
                kpt_heatmap = F.interpolate(
                    kpt_heatmap,
                    scale_factor=2,
                    mode='bicubic',
                    align_corners=False)
                kpt_heatmaps.append(kpt_heatmap)
                # self.show_keypoint_heatmap(kpt_heatmap, 8, img, img_meta)
            kpt_heatmaps = torch.cat(kpt_heatmaps)
            # kpt_heatmaps = kpt_heatmaps.max(dim=0)[0]
            # self.show_keypoint_heatmap(kpt_heatmaps[None], 8, img, img_meta)
            pos = kpt_heatmaps.reshape(
                kpt_heatmaps.size(0), kpt_heatmaps.size(1), -1).argmax(dim=2)
            x_int = pos % kpt_heatmaps.size(3)
            y_int = (pos - x_int) // kpt_heatmaps.size(3)

            # compensate the downsampling error empirically
            # x = (x_int.float() + 0.5) * 4
            # y = (y_int.float() + 0.5) * 4
            # compensate the downsampling error by disk_offset regression
            K, J = det_keypoints.shape[:2]
            if self.hm_offset_feat.size(1) == 2:
                hp_offset = transpose_and_gather_feat(
                    self.hm_offset_feat[0],
                    pos.permute(1, 0).view(-1))  # JK x 2
                hp_offset = hp_offset.view(J, K, 2)
                x = (x_int.float() + hp_offset.permute(1, 0)[:, :, 0]) * 4
                y = (y_int.float() + hp_offset.permute(1, 0)[:, :, 1]) * 4
            elif self.hm_offset_feat.size(1) == 34:
                hp_offset = []
                for j in range(J):
                    hp_offset_j = transpose_and_gather_feat(
                        self.hm_offset_feat[0, j * 2:j * 2 + 2],
                        pos.permute(1, 0)[j])
                    hp_offset.append(hp_offset_j)
                hp_offset = torch.cat(hp_offset)
                hp_offset = hp_offset.view(J, K, 2)
                x = (x_int.float() + hp_offset.permute(1, 0, 2)[:, :, 0]) * 4
                y = (y_int.float() + hp_offset.permute(1, 0, 2)[:, :, 1]) * 4

            kpt_loc = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=2)
            if rescale:
                kpt_loc /= mlvl_bboxes.new_tensor(scale_factor[:2])
            det_keypoints[..., :2] = kpt_loc
        return det_bboxes, det_labels, det_keypoints

    def get_pseudo_bbox(self, kpts):
        kpts_x = kpts[:, :, 0]
        kpts_y = kpts[:, :, 1]
        x1 = kpts_x.min(dim=1, keepdim=True)[0]
        y1 = kpts_y.min(dim=1, keepdim=True)[0]
        x2 = kpts_x.max(dim=1, keepdim=True)[0]
        y2 = kpts_y.max(dim=1, keepdim=True)[0]
        bboxes = torch.cat([x1, y1, x2, y2], dim=1)
        return bboxes

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list,
                    gt_keypoints_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, inds_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_keypoints_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        inds_list = [inds.split(num_points, 0) for inds in inds_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_inds = []
        concat_lvl_imgs = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_inds.append(torch.cat([inds[i] for inds in inds_list]))
            img_ind = []
            for j, labels in enumerate(labels_list):
                img_ind.extend([j for _ in range(labels[i].size(0))])
            img_ind = concat_lvl_inds[i].new_tensor(img_ind)
            concat_lvl_imgs.append(img_ind)
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_inds, concat_lvl_imgs

    def _get_target_single(self, gt_bboxes, gt_labels, gt_keypoints, points,
                           regress_ranges, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels.new_zeros(num_points,)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left,
                cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, min_area_inds

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def keypoint_target(self, keypoints, points, strides):
        offset_targets = keypoints - points[:, None, :].expand(keypoints.shape)
        offset_targets /= strides[:, None, None]
        return offset_targets

    def heatmap_ae_offset_loss(self, hm_pred, ae_pred, hm_offset_pred,
                               gt_keypoints, gt_labels, gt_bboxes, img_metas):
        assert hm_pred.shape[-2:] == hm_offset_pred.shape[-2:]
        num_img, _, h, w = hm_pred.size()
        # placeholder of heatmap target (Gaussian distribution)
        hm_target = hm_pred.new_zeros(hm_pred.shape)
        # placeholder of centripetal offset target and mask
        hm_offset_target = hm_offset_pred.new_ones(hm_offset_pred.shape) * INF
        hm_offset_mask = hm_offset_pred.new_zeros(hm_offset_pred.shape)
        loss_ae_pull = 0
        loss_ae_push = 0
        for i, (gt_label, gt_bbox, gt_keypoint) in enumerate(
                zip(gt_labels, gt_bboxes, gt_keypoints)):
            if gt_label.size(0) == 0:
                continue
            gt_keypoint = gt_keypoint.reshape(gt_keypoint.shape[0], -1,
                                              3).clone()
            gt_keypoint[..., :2] /= 4
            # x_coord
            assert gt_keypoint[..., 0].max() <= w
            # y_coord
            assert gt_keypoint[..., 1].max() <= h
            gt_bbox /= 4
            gt_w = gt_bbox[:, 2] - gt_bbox[:, 0]
            gt_h = gt_bbox[:, 3] - gt_bbox[:, 1]
            tags = []
            pull, push = 0, 0
            for j in range(gt_label.size(0)):
                kp_radius = torch.clamp(
                    torch.floor(
                        gaussian_radius((gt_h[j], gt_w[j]),
                                        min_overlap=self.min_overlap_hm)),
                    min=self.min_hm_radius,
                    max=self.max_hm_radius)
                offset_radius = torch.clamp(
                    torch.floor(
                        gaussian_radius((gt_h[j], gt_w[j]),
                                        min_overlap=self.min_overlap_kp)),
                    min=self.min_offset_radius,
                    max=self.max_offset_radius)
                tmp = []
                for k in range(self.num_keypoints):
                    if gt_keypoint[j, k, 2] > 0:
                        gt_kp = gt_keypoint[j, k, :2]
                        gt_kp_int = torch.floor(gt_kp)
                        draw_umich_gaussian(hm_target[i, k], gt_kp_int,
                                            kp_radius)
                        draw_short_range_offset(
                            hm_offset_target[i, k * 2:k * 2 + 2, :, :],
                            hm_offset_mask[i, k * 2:k * 2 + 2, :, :], gt_kp,
                            offset_radius)
                        tmp.append(ae_pred[i, k, gt_kp_int[1].long() - 1,
                                           gt_kp_int[0].long() - 1])
                if len(tmp) == 0:
                    continue
                tmp = torch.stack(tmp)
                tags.append(torch.mean(tmp))
                pull += torch.mean((tmp - tags[-1]) ** 2)
            num_tags = len(tags)
            if num_tags == 0:
                pull = push = ae_pred[i].sum() * 0.0
            elif num_tags == 1:
                push = ae_pred[i].sum() * 0.0
            else:
                tags = torch.stack(tags)
                tags_expand = tags.expand(num_tags, num_tags)
                diff = tags_expand - tags_expand.permute(1, 0)
                if self.ae_loss_type == 'exp':
                    diff = torch.pow(diff, 2)
                    push = torch.exp(-diff)
                    push = torch.sum(push) - num_tags
                elif self.ae_loss_type == 'max':
                    diff - 1 - torch.abs(diff)
                    push = torch.clamp(diff, min=0).sum() - num_tags
                else:
                    raise ValueError('Unknown as loss type')
                pull = pull / num_tags
                push = push / (num_tags * (num_tags - 1) * 2)
            loss_ae_pull += self.ae_loss_weight * pull / num_img
            loss_ae_push += self.ae_loss_weight * push / num_img
        # compute heatmap loss
        # refer to CenterNet
        hm_pred = torch.clamp(hm_pred.sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss_hm = self.loss_hm(hm_pred, hm_target)
        # compute offset loss
        hm_offset_target[hm_offset_target == INF] = 0
        loss_hm_offset = F.l1_loss(
            hm_offset_pred * hm_offset_mask,
            hm_offset_target * hm_offset_mask,
            reduction='sum')
        loss_hm_offset = loss_hm_offset / (hm_offset_mask.sum() + 1e-4)
        if not torch.is_tensor(loss_ae_pull):
            loss_ae_pull = loss_ae_push = ae_pred.sum() * 0.0
        return loss_hm, loss_ae_pull, loss_ae_push, loss_hm_offset

    def show_keypoint_heatmap(self, hm_feat, scale_factor, img, img_meta):
        import os.path as osp
        import mmcv
        from mmcv.image import tensor2imgs, imread, imwrite
        h, w, _ = img_meta['img_shape']
        img_show = tensor2imgs(img.unsqueeze(0),
                               **img_meta['img_norm_cfg'])[0][:h, :w, :]
        hm_feat = F.interpolate(
            hm_feat,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False)[:, :, :h, :w].permute(2, 3, 1, 0)
        colors_kp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
                     (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
                     (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
                     (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
                     (0, 0, 255)]
        colors_kp = img.new_tensor(colors_kp).reshape(1, 1, -1, 3)
        colors_kp = 255 - colors_kp
        color_map = (hm_feat * colors_kp).max(dim=2)[0].permute(2, 0,
                                                                1).unsqueeze(0)
        color_map = tensor2imgs(color_map, to_rgb=False)[0]
        color_map = 255 - color_map
        color_map = img_show * 0.3 + color_map * 0.7
        color_map[color_map > 255] = 255
        color_map[color_map < 0] = 0
        color_map = color_map.astype(np.uint8)
        mmcv.imshow(color_map)
        # imwrite(color_map, osp.join('../../checkpoints/res',
        #     osp.splitext(osp.basename(img_meta['filename']))[0] + '_hm.jpg'))

    def get_coord_map(self, feat, point, normalize):
        h, w = feat.shape[2:]
        x, y = point
        stride = self.strides[0]
        x_coord = torch.arange(0, w * stride, step=stride,
            dtype=torch.float32, device=feat.device) + stride // 2
        y_coord = torch.arange(0, h * stride, step=stride,
            dtype=torch.float32, device=feat.device) + stride // 2
        x_coord = x_coord - x
        y_coord = y_coord - y
        x_coord = x_coord / normalize
        y_coord = y_coord / normalize
        y_map, x_map = torch.meshgrid(y_coord, x_coord)
        return torch.cat([x_map[None], y_map[None]], dim=0)

    def kpt_fcn_head(self, weight, x, coord_map):
        conv1_weight = weight[:80].view(8, 10, 1, 1)
        conv1_bias = weight[80:88]
        conv2_weight = weight[88:152].view(8, 8, 1, 1)
        conv2_bias = weight[152:160]
        conv3_weight = weight[160:296].view(17, 8, 1, 1)
        conv3_bias = weight[296:]
        relu = nn.ReLU(inplace=True)
        x = torch.cat([x, coord_map[None]], dim=1)
        x = F.conv2d(x, conv1_weight, conv1_bias)
        x = relu(x)
        x = F.conv2d(x, conv2_weight, conv2_bias)
        x = relu(x)
        x = F.conv2d(x, conv3_weight, conv3_bias)
        return x
