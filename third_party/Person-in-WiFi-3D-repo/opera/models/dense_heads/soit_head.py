# Copyright (c) Hikvision Research Institute. All rights reserved.
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.dense_heads.detr_head import DETRHead

from opera.models.utils import build_positional_encoding
from ..builder import HEADS


@HEADS.register_module()
class SOITHead(DETRHead):
    """Head of SOIT: Segmenting Objects with Instance-Aware Transformers.

    More details can be found in the `paper
    <https://arxiv.org/abs/2112.11037>`_ .

    Args:
        dynamic_params_dims (int): Number of dynamic parameters.
        dynamic_encoder_heads (int): Number of multi-heads in dynamic encoder.
        mask_positional_encoding_cfg (obj:`ConfigDict`): ConfigDict is used
            for building positional encoding for mask feature.
        dice_mask_loss_weight (float): Loss weight of dice mask loss.
        bce_mask_loss_weight (float): Loss weight of bce mask loss.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Transformer.
    """

    def __init__(self,
                 *args,
                 num_seg_fcs=2,
                 dynamic_params_dims=441,
                 dynamic_encoder_heads=4,
                 mask_positional_encoding_cfg=dict(
                     type='RelSinePositionalEncoding',
                     num_feats=4,
                     normalize=True),
                 dice_mask_loss_weight=1.0,
                 bce_mask_loss_weight=1.0,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_seg_fcs = num_seg_fcs
        self.dynamic_params_dims = dynamic_params_dims
        self.dynamic_encoder_heads = dynamic_encoder_heads
        self.mask_positional_encoding_cfg = mask_positional_encoding_cfg
        self.dice_mask_loss_weight = dice_mask_loss_weight
        self.bce_mask_loss_weight = bce_mask_loss_weight

        super(SOITHead, self).__init__(
            *args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize classification branch, regression branch and
        segmentation branch of head.
        """
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        seg_branch = []
        for _ in range(self.num_seg_fcs):
            seg_branch.append(Linear(self.embed_dims, self.embed_dims))
            seg_branch.append(nn.ReLU())
        seg_branch.append(Linear(self.embed_dims, self.dynamic_params_dims))
        seg_branch = nn.Sequential(*seg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        self.seg_branches = nn.ModuleList(
            [seg_branch for _ in range(num_pred - 1)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

        self.dynamic_encoder = DynamicDeformableAttention(
            embed_dims=self.transformer.mask_channels,
            num_heads=self.dynamic_encoder_heads)
        self.mask_positional_encoding = build_positional_encoding(
            self.mask_positional_encoding_cfg)

    def init_weights(self):
        """Initialize weights of the prediction head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image.
            gt_labels (list[Tensor]): Ground truth labels of bboxes in each image.
            gt_masks (list[Tensor]): Ground truth segmentation masks of each image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be ignored of each image.
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should include background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is Ture it would be returned, otherwise \
                `None` would be returned.
            mask_proto (Tuple[Tensor]): Mask feature and other information 
                to create the input for dynamic encoder.
            outputs_dynamic_params (Tensor): Dynamic parameters of all decoder
                layers, has shape [nb_dec, bs, num_query, dynamic_params_dims].
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        p3_mask = F.interpolate(
            img_masks[None],
            size=mlvl_feats[0].shape[-2:]).to(torch.bool).squeeze(0)
        self.p3_mask = p3_mask

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, mask_proto = \
                self.transformer(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    reg_branches=self.reg_branches \
                        if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches \
                        if self.as_two_stage else None  # noqa:E501
            )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_dynamic_params = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            dynamic_params = self.seg_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_dynamic_params.append(dynamic_params)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_dynamic_params = torch.stack(outputs_dynamic_params)
        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class, \
                enc_outputs_coord.sigmoid(), mask_proto, outputs_dynamic_params
        else:
            return outputs_classes, outputs_coords, \
                None, None, mask_proto, outputs_dynamic_params

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds,
             mask_proto,
             dynamic_params,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all decoder
                layers, has shape [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            mask_proto (Tuple(Tensor)): Mask feature and other information 
                to create the input for dynamic encoder.
            dynamic_params (Tensor): Dynamic parameters of all decoder layers,
                has shape [nb_dec, bs, num_query, dynamic_params_dims].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth segmentation masks for
                each image with shape (num_gts, img_h, img_w).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        losses_mask_dice, losses_mask_bce = multi_apply(
            self.loss_mask_single, dynamic_params, all_cls_scores,
            all_bbox_preds, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list,
            mask_proto=mask_proto)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_mask_dice'] = losses_mask_dice[-1]
        loss_dict['loss_mask_bce'] = losses_mask_bce[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_mask_dice_i, \
            loss_mask_bce_i in zip(losses_cls[:-1], losses_bbox[:-1], \
                losses_iou[:-1], losses_mask_dice[:-1], losses_mask_bce[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_mask_dice'] = loss_mask_dice_i
            loss_dict[f'd{num_dec_layer}.loss_mask_bce'] = loss_mask_bce_i
            num_dec_layer += 1

        return loss_dict

    def loss_mask_single(self,
                         dynamic_params,
                         cls_scores,
                         bbox_preds,
                         gt_bboxes_list,
                         gt_labels_list,
                         gt_masks_list,
                         img_metas,
                         gt_bboxes_ignore_list,
                         mask_proto=None):
        (seg_memory, seg_pos_embed, seg_mask, spatial_shapes,
        seg_reference_points, level_start_index, valid_ratios) = mask_proto
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets_mask(cls_scores_list, bbox_preds_list,
                                                gt_bboxes_list, gt_labels_list,
                                                img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, gt_inds_list) = cls_reg_targets

        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        gt_masks_list = [gt_masks.masks for gt_masks in gt_masks_list]
        input_h, input_w = img_metas[0]['batch_input_shape']
        loss_mask_dice = 0
        loss_mask_bce = 0
        for i in range(num_imgs):
            mask_h, mask_w = gt_masks_list[i].shape[1:]
            gt_masks = nn.ConstantPad2d(
                (0, input_w - mask_w, 0, input_h - mask_h), 0) \
                    (seg_memory.new_tensor(gt_masks_list[i]))
            gt_inds = gt_inds_list[i]
            pos_ind = gt_inds > 0
            gt_masks = gt_masks[gt_inds[pos_ind] - 1]
            mask_preds = []
            pos_dynamic_params = dynamic_params[i][pos_ind]
            pos_bbox_preds = bbox_preds[i][pos_ind].detach()
            pos_cxcy_coord = pos_bbox_preds[:, :2]
            img_mask = self.p3_mask[[i]]

            if pos_dynamic_params.size(0) > 0:
                for j in range(pos_dynamic_params.size(0)):
                    seg_pos_embed = self.mask_positional_encoding(
                        img_mask, pos_cxcy_coord[j])
                    seg_pos_embed = seg_pos_embed.flatten(2).transpose(
                        1, 2).permute(1, 0, 2)
                    mask_preds.append(self.dynamic_encoder(
                        pos_dynamic_params[j],
                        seg_memory[:, [i], :],
                        None,
                        None,
                        query_pos=seg_pos_embed,
                        key_padding_mask=seg_mask[[i]],
                        reference_points=seg_reference_points[[i]],
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index))
                h, w = spatial_shapes[0]
                mask_preds = [
                    mask.squeeze().reshape(h, w) for mask in mask_preds]
                mask_preds = torch.stack(mask_preds)
                pad_mask = seg_mask[i].reshape(1, 1, h, w).float()
                pad_mask = F.interpolate(
                    pad_mask,
                    scale_factor=4,
                    mode='bilinear',
                    align_corners=True).to(torch.bool).squeeze()
                mask_preds = aligned_bilinear(
                    mask_preds[None], factor=4)[0].sigmoid()
                mask_preds.masked_fill(pad_mask, 0)
                mask_targets = F.interpolate(
                    gt_masks[None],
                    size=(4*h, 4*w),
                    mode='bilinear',
                    align_corners=True)[0]
                loss_mask_dice += self.dice_loss(mask_preds,
                                                 mask_targets).sum()
                loss_mask_bce += F.binary_cross_entropy(
                    mask_preds,
                    mask_targets,
                    reduction='none').sum() / (~pad_mask).sum()
            else:
                loss_mask_dice += (pos_dynamic_params.sum() * 
                    seg_memory[:, i, :].sum() * seg_pos_embed.sum())
                loss_mask_bce += (pos_dynamic_params.sum() * 
                    seg_memory[:, i, :].sum() * seg_pos_embed.sum())

        loss_mask_dice = (loss_mask_dice / num_total_pos) * \
            self.dice_mask_loss_weight
        loss_mask_bce = (loss_mask_bce / num_total_pos) * \
            self.bce_mask_loss_weight
        
        return loss_mask_dice, loss_mask_bce

    def get_targets_mask(self,
                         cls_scores_list,
                         bbox_preds_list,
                         gt_bboxes_list,
                         gt_labels_list,
                         img_metas,
                         gt_bboxes_ignore_list=None):
        """Compute segmentation targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list, gt_inds_list) = multi_apply(
            self._get_target_single_mask, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, gt_inds_list)

    def _get_target_single_mask(self,
                                cls_score,
                                bbox_pred,
                                gt_bboxes,
                                gt_labels,
                                img_meta,
                                gt_bboxes_ignore=None):
        """Compute segmentation targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, assign_result.gt_inds)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   mask_proto,
                   dynamic_params,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox and mask predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all decoder
                layers, has shape [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of points on
                encode feature map, has shape (N, h*w, num_classes).
                Only be passed when as_two_stage is True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            mask_proto (Tuple(Tensor)): Mask feature and other information 
                to create the input for dynamic encoder.
            dynamic_params (Tensor): Dynamic parameters of all decoder layers,
                has shape [nb_dec, bs, num_query, dynamic_params_dims].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Defalut False.

        Returns:
            list[list[Tensor, Tensor, Tensor]]: Each item in result_list is 3-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box. The last item is a (n, h, w) tensor \
                where each element is the predicted mask of the corresponding instance.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        dynamic_params = dynamic_params[-1]
        (seg_memory, seg_pos_embed, seg_mask, spatial_shapes,
         seg_reference_points, level_start_index, valid_ratios) = mask_proto
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            dynamic_param = dynamic_params[img_id]
            single_seg_memory = seg_memory[:, [img_id], :]
            single_seg_pos_embed = seg_pos_embed[:, [img_id], :]
            single_seg_mask = seg_mask[[img_id]]
            single_seg_reference_points = seg_reference_points[[img_id]]
            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, dynamic_param, single_seg_memory,
                single_seg_pos_embed, single_seg_mask,
                single_seg_reference_points, level_start_index, spatial_shapes,
                img_shape, ori_shape, scale_factor, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           dynamic_params,
                           seg_memory,
                           seg_pos_embed,
                           seg_mask,
                           seg_reference_points,
                           level_start_index,
                           spatial_shapes,
                           img_shape,
                           ori_shape,
                           scale_factor,
                           rescale=False):
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        mask_thr = self.test_cfg.get('mask_thresh', 0.5)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexs = cls_score.view(-1).topk(max_per_img)
            det_labels = indexs % self.num_classes
            bbox_index = indexs // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        dynamic_params = dynamic_params[bbox_index]
        cxcy = bbox_pred[:, :2]
        img_mask = self.p3_mask
        seg_pos_embeds = []
        num_res = dynamic_params.size(0)
        for i in range(dynamic_params.size(0)):
            seg_pos_embed = self.mask_positional_encoding(img_mask, cxcy[i])
            seg_pos_embed = \
                seg_pos_embed.flatten(2).transpose(1, 2).permute(1, 0, 2)
            seg_pos_embeds.append(seg_pos_embed)
        seg_pos_embeds = torch.cat(seg_pos_embeds, dim=1)
        seg_memory = seg_memory.repeat(1, num_res, 1)
        seg_reference_points = seg_reference_points.repeat(num_res, 1, 1, 1)
        mask_preds = self.dynamic_encoder.forward_test(
            dynamic_params,
            seg_memory,
            None,
            None,
            query_pos=seg_pos_embeds,
            key_padding_mask=seg_mask,
            reference_points=seg_reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        h, w = spatial_shapes[0]
        mask_preds = mask_preds.squeeze().reshape(num_res, h, w)

        pad_mask = seg_mask.reshape(1, 1, h, w).float()
        pad_mask = F.interpolate(
            pad_mask,
            scale_factor=4,
            mode='bilinear',
            align_corners=True).to(torch.bool).squeeze()
        mask_preds = aligned_bilinear(mask_preds[None], factor=4).sigmoid()
        mask_preds.masked_fill(pad_mask, 0)
        det_masks = F.interpolate(
            mask_preds,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=True)
        if rescale:
            det_masks = F.interpolate(
                det_masks,
                size=ori_shape[:2],
                mode='bilinear',
                align_corners=True)[0]
        else:
            det_masks = det_masks[0]
        det_masks = det_masks > mask_thr
        # BG is not included in num_classes
        cls_segms = [[] for _ in range(self.num_classes)]
        N = det_masks.size(0)
        for i in range(N):
            cls_segms[det_labels[i]].append(det_masks[i].detach().cpu().numpy())

        return det_bboxes, det_labels, cls_segms

    def dice_loss(self, pred, target):
        smooth = 1e-5
        num = pred.size(0)
        if num == 0:
            return pred.sum(-1).sum(-1)
        iflat = pred.view(num, -1)
        tflat = target.view(num, -1)
        intersection = iflat * tflat

        return 1 - (2. * intersection.sum(1) /
              ((iflat * iflat).sum(1) + (tflat * tflat).sum(1) + smooth))


class DynamicDeformableAttention(BaseModule):
    """A dynamic attention module used in SOIT. The parameters of this module
    are generated from transformer decoder head.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0.1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=1,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.im2col_step_test = 100  # for test
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

    def forward(self,
                dynamic_params,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            dynamic_params (Tensor): Dynamic generated parameters for 
                MultiScaleDeformAttention with shape (dynamic_params_dims).
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_key

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_key, self.num_heads, -1)
        #split the dynamic parameters to each layer
        sampling_offsets_weight = dynamic_params[:256].reshape(32, 8)
        sampling_offsets_bias = dynamic_params[256:288]
        attention_weights_weight = dynamic_params[288:416].reshape(16, 8)
        attention_weights_bias = dynamic_params[416:432]
        output_proj_weights = dynamic_params[432:440].reshape(1, 8)
        output_proj_bias = dynamic_params[440]
        sampling_offsets = F.linear(
            query, sampling_offsets_weight, sampling_offsets_bias).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = F.linear(
            query, attention_weights_weight, attention_weights_bias).view(
                bs, num_query, self.num_heads, self.num_levels * self.num_points)
 
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)

        output = output.relu()     
        output = F.linear(
            output, output_proj_weights, output_proj_bias).permute(1, 0, 2)
        return output

    def forward_test(self,
                     dynamic_params,
                     query,
                     key,
                     value,
                     residual=None,
                     query_pos=None,
                     key_padding_mask=None,
                     reference_points=None,
                     spatial_shapes=None,
                     level_start_index=None,
                     **kwargs):
        """Faster version for dynamic encoder inference"""

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_key

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_key, self.num_heads, -1)

        query = query.permute(0, 2, 1).reshape(1, -1, num_query)
        sampling_offsets_weight = dynamic_params[:, :256].reshape(bs * 32, 8, 1)
        sampling_offsets_bias = dynamic_params[:, 256:288].reshape(bs * 32)
        attention_weights_weight = dynamic_params[:, 288:416].reshape(bs * 16, 8, 1)
        attention_weights_bias = dynamic_params[:, 416:432].reshape(bs * 16)
        output_proj_weights = dynamic_params[:, 432:440].reshape(bs * 1, 8, 1)
        output_proj_bias = dynamic_params[:, 440].reshape(bs * 1)
        sampling_offsets = F.conv1d(
            query, sampling_offsets_weight, sampling_offsets_bias, \
                groups=bs).view(bs, self.num_heads, self.num_levels, \
                    self.num_points, 2, num_query).permute(
                        0, 5, 1, 2, 3, 4).contiguous()
        attention_weights = F.conv1d(
            query, attention_weights_weight, attention_weights_bias, \
                groups=bs).view(bs, self.num_heads, \
                    self.num_levels * self.num_points, num_query).permute(
                        0, 3, 1, 2).contiguous()

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step_test)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step_test)
        output = output.relu()

        output = output.permute(0, 2, 1).reshape(1, -1, num_query)
        output = F.conv1d(output,
                          output_proj_weights,
                          output_proj_bias,
                          groups=bs).permute(1, 0, 2)
        return output


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]
