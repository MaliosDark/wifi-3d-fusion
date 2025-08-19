# Copyright (c) Hikvision Research Institute. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.nn.init import normal_
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence)
from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttention,
                                              MultiScaleDeformableAttnFunction,
                                              multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.transformer import (DeformableDetrTransformer,
                                            Transformer, inverse_sigmoid)

from .builder import (TRANSFORMER, ATTENTION, TRANSFORMER_LAYER_SEQUENCE,
                      build_transformer_layer_sequence)


@ATTENTION.register_module()
class MultiScaleDeformablePoseAttention(BaseModule):
    """An attention module used in PETR. `End-to-End Multi-Person
    Pose Estimation with Transformers`.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 17.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0.1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=17,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

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
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
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
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape (num_key, bs, embed_dims).
            value (Tensor): The value tensor with shape
                (num_key, bs, embed_dims).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference points with
                shape (bs, num_query, num_levels, K*2), all elements is range
                in [0, 1], top-left (0,0), bottom-right (1, 1), including
                padding area.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
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
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_key

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_key, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == self.num_points * 2:
            reference_points_reshape = reference_points.reshape(
                bs, num_query, self.num_levels, -1, 2).unsqueeze(2)
            x1 = reference_points[:, :, :, 0::2].min(dim=-1, keepdim=True)[0]
            y1 = reference_points[:, :, :, 1::2].min(dim=-1, keepdim=True)[0]
            x2 = reference_points[:, :, :, 0::2].max(dim=-1, keepdim=True)[0]
            y2 = reference_points[:, :, :, 1::2].max(dim=-1, keepdim=True)[0]
            w = torch.clamp(x2 - x1, min=1e-4)
            h = torch.clamp(y2 - y1, min=1e-4)
            wh = torch.cat([w, h], dim=-1)[:, :, None, :, None, :]

            sampling_locations = reference_points_reshape \
                                 + sampling_offsets * wh * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2K, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        output = self.output_proj(output).permute(1, 0, 2)
        # (num_query, bs ,embed_dims)
        return self.dropout(output) + inp_residual


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in PETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 return_intermediate=False,
                 num_keypoints=17,
                 **kwargs):

        super(PetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_keypoints = num_keypoints

    def forward(self,
                query,
                *args,
                reference_points=None,
                kpt_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            reference_points (Tensor): The reference points of offset,
                has shape (bs, num_query, K*2).
            valid_ratios (Tensor): The radios of valid points on the feature
                map, has shape (bs, num_levels, 2).
            kpt_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results. Only would be passed when `with_box_refine`
                is True, otherwise would be passed a `None`.

        Returns:
            tuple (Tensor): Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims] and
                [num_layers, bs, num_query, K*2].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                *args,
                **kwargs)
            output = output.permute(1, 0, 2)

            if kpt_branches is not None:
                tmp = kpt_branches[lid](output)
                if reference_points.shape[-1] == self.num_keypoints * 3:
                    new_reference_points = tmp + reference_points
                    new_reference_points = new_reference_points
                else:
                    raise NotImplementedError
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PetrRefineTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(PetrRefineTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            assert reference_points.shape[-1] == 3
            output = layer(
                output,
                *args,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3
                new_reference_points = tmp
                new_reference_points[..., :3] = tmp[
                    ..., :3] + reference_points
                new_reference_points = new_reference_points
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class PETRTransformer(Transformer):
    """Implements the PETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 refine_decoder=dict(
                     type='DeformableDetrTransformerDecoder',
                     num_layers=1,
                     return_intermediate=True,
                     transformerlayers=dict(
                         type='DetrTransformerDecoderLayer',
                         attn_cfgs=[
                             dict(
                                 type='MultiheadAttention',
                                 embed_dims=256,
                                 num_heads=8,
                                 dropout=0.1),
                             dict(
                                 type='MultiScaleDeformableAttention',
                                 embed_dims=256)
                         ],
                         feedforward_channels=1024,
                         ffn_dropout=0.1,
                         operation_order=('self_attn', 'norm', 'cross_attn',
                                          'norm', 'ffn', 'norm'))),
                 as_two_stage=True,
                 num_feature_levels=4,
                 two_stage_num_proposals=100,
                 num_keypoints=17,
                 **kwargs):
        super(PETRTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.num_keypoints = num_keypoints
        self.init_layers()
        self.refine_decoder = build_transformer_layer_sequence(refine_decoder)

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.refine_query_embedding = nn.Embedding(self.num_keypoints,
                                                       self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims,
                                              2 * self.num_keypoints)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        for m in self.modules():
            if isinstance(m, MultiScaleDeformablePoseAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        normal_(self.refine_query_embedding.weight)


    def forward(self,
                mlvl_feats,
                query_embed,
                kpt_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from different level.
                Each element has shape [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from different
                level used for encoder and decoder, each element has shape
                [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            kpt_branches (obj:`nn.ModuleList`): Keypoint Regression heads for
                feature maps from each decoder layer. Only would be passed when
                `with_box_refine` is Ture. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads for
                feature maps from each decoder layer. Only would be passed when
                `as_two_stage` is Ture. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    `return_intermediate_dec` is True output has shape \
                    (num_dec_layers, bs, num_query, embed_dims), else has \
                    shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of proposals \
                    generated from encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_kpt_unact: The regression results generated from \
                    encoder's feature maps., has shape (batch, h*w, K*2).
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None
        feat_flatten = mlvl_feats.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        if self.as_two_stage:
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                memory)
            enc_outputs_kpt_unact = \
                kpt_branches[self.decoder.num_layers](memory)

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            # topk_coords_unact = torch.gather(
            #     enc_outputs_coord_unact, 1,
            #     topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            # topk_coords_unact = topk_coords_unact.detach()
            topk_kpts_unact = torch.gather(
                enc_outputs_kpt_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_kpt_unact.size(-1)))
            topk_kpts_unact = topk_kpts_unact.detach()

            reference_points = topk_kpts_unact
            init_reference_out = reference_points
            # learnable query and query_pos
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            # key=None,
            key=memory,
            value=memory,
            query_pos=query_pos,
            reference_points=reference_points,
            kpt_branches=kpt_branches,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, \
                   inter_references_out, enc_outputs_class, \
                   enc_outputs_kpt_unact, memory
        return inter_states, init_reference_out, \
               inter_references_out, None, None, None, None, None

    def forward_refine(self,
                       memory,
                       reference_points_pose,
                       img_inds,
                       kpt_branches=None,
                       **kwargs):

        # pose refinement (17 queries corresponding to 17 keypoints)
        # learnable query and query_pos
        refine_query_embedding = self.refine_query_embedding.weight
        query_pos, query = torch.split(
            refine_query_embedding, refine_query_embedding.size(1) // 2, dim=1)
        pos_num = reference_points_pose.size(0)
        query_pos = query_pos.unsqueeze(0).expand(pos_num, -1, -1)
        query = query.unsqueeze(0).expand(pos_num, -1, -1)
        reference_points = reference_points_pose.reshape(
            pos_num,
            reference_points_pose.size(1) // 3, 3)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        pos_memory = memory[:, img_inds, :]

        inter_states, inter_references = self.refine_decoder(
            query=query,
            key=pos_memory,
            value=pos_memory,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=kpt_branches,
            **kwargs)
        # [num_decoder, num_query, bs, embed_dim]

        init_reference_out = reference_points
        return inter_states, init_reference_out, inter_references
