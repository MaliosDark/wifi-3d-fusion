_base_ = [
    '../_base_/datasets/crowdpose_keypoint.py',
    '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'  # noqa
model = dict(
    type='opera.PETR',
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.PETRHead',
        num_query=300,
        num_keypoints=14,
        num_classes=1,  # only person
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='opera.PETRTransformer',
            num_keypoints=14,
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.PetrTransformerDecoder',
                num_layers=6,
                num_keypoints=14,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='opera.MultiScaleDeformablePoseAttention',
                            embed_dims=256,
                            num_points=14)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            hm_encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            refine_decoder=dict(
                type='mmcv.DeformableDetrTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiScaleDeformableAttention',
                            embed_dims=256,
                            im2col_step=128)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_kpt=dict(type='mmdet.L1Loss', loss_weight=70.0),
        loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
        loss_oks=dict(type='opera.OKSLoss', loss_weight=2.0, num_keypoints=14),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=4.0),
        loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=80.0),
        loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=3.0, 
                             num_keypoints=14)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', weight=7.0,
                          num_keypoints=14))),
    test_cfg=dict(max_per_img=100))  # set 'max_per_img=20' for time counting
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
