dataset_type = 'opera.WifiPoseDataset'
data_root = '/home/yankangwei/opera-main/data/wifipose'
train_pipeline = [
    dict(
        type='opera.DefaultFormatBundle',
        extra_keys=['gt_keypoints', 'gt_labels']),
    dict(
        type='mmdet.Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'],
        meta_keys=[])
]
test_pipeline = [
    dict(
        type='mmdet.MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(
                type='opera.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_labels']),
            dict(type='mmdet.Collect', keys=['img'], meta_keys=[])
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='opera.WifiPoseDataset',
        dataset_root='/home/yankangwei/opera-main/data/wifipose/train_data',
        pipeline=[
            dict(
                type='opera.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_labels']),
            dict(
                type='mmdet.Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'
                ],
                meta_keys=[])
        ],
        mode='train'),
    val=dict(
        type='opera.WifiPoseDataset',
        dataset_root='/home/yankangwei/opera-main/data/wifipose/test_data',
        pipeline=[
            dict(
                type='mmdet.MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(
                        type='opera.DefaultFormatBundle',
                        extra_keys=['gt_keypoints', 'gt_labels']),
                    dict(type='mmdet.Collect', keys=['img'], meta_keys=[])
                ])
        ],
        mode='test'),
    test=dict(
        type='opera.WifiPoseDataset',
        dataset_root='/home/yankangwei/opera-main/data/wifipose/test_data',
        pipeline=[
            dict(
                type='mmdet.MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(
                        type='opera.DefaultFormatBundle',
                        extra_keys=['gt_keypoints', 'gt_labels']),
                    dict(type='mmdet.Collect', keys=['img'], meta_keys=[])
                ])
        ],
        mode='test'))
evaluation = dict(interval=1, metric='mpjpe')
checkpoint_config = dict(interval=1, max_keep_ckpts=20)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(
    type='opera.PETR',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.PETRHead',
        num_query=100,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        num_keypoints=14,
        transformer=dict(
            type='opera.PETRTransformer',
            num_keypoints=14,
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.PetrTransformerDecoder',
                num_layers=3,
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
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            refine_decoder=dict(
                type='opera.PetrRefineTransformerDecoder',
                num_layers=2,
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
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
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
            loss_weight=4.0),
        loss_kpt=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_kpt_rpn=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_oks=dict(type='opera.OKSLoss', loss_weight=2.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=4.0),
        loss_kpt_refine=dict(type='mmdet.MSELoss', loss_weight=70.0),
        loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=3.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=4.0),
            kpt_cost=dict(type='opera.KptMSECost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', weight=7.0))),
    test_cfg=dict(max_per_img=100))
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[400])
runner = dict(type='EpochBasedRunner', max_epochs=450)
find_unused_parameters = True
work_dir = '/home/yankangwei/opera-main/result/wifipose'
auto_resume = False
gpu_ids = range(0, 3)
