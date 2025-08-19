_base_ = [
    '../_base_/default_runtime.py'
]
model = dict(
    type='opera.InsPose',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='opera.InsPoseHead',
        num_classes=1, # (only person)
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        stacked_convs_kpt=4,
        feat_channels_kpt=512,
        stacked_convs_hm=3,
        feat_channels_hm=512,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        centerness_on_reg=True,
        regression_normalize=True,
        with_hm_loss=True,
        min_overlap_hm=0.9,
        min_hm_radius=0,
        max_hm_radius=3,
        min_overlap_kp=0.9,
        min_offset_radius=0,
        max_offset_radius=3,
        loss_cls=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.75,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=1.0),
        loss_weight_offset=1.0,
        unvisible_weight=0.1),
    test_cfg=dict(
        nms_pre=1000,
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.3),
        mask_thresh=0.5,
        max_per_img=100))
# dataset settings
dataset_type = 'opera.CocoPoseDataset'
data_root = '/dataset/public/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', to_float32=True),
    dict(type='opera.LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         with_keypoint=True),
    dict(type='opera.Resize',
         img_scale=[(1333, 800), (1333, 768), (1333, 736),
                    (1333, 704), (1333, 672), (1333, 640)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='opera.RandomFlip', flip_ratio=0.5),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=32),
    dict(type='opera.DefaultFormatBundle', extra_keys=['gt_keypoints']),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_keypoints']),
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/data/configs/inspose/person_keypoints_train2017_pseudobox.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline,
        skip_invaild_pose=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/person_keypoints_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='keypoints')
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
