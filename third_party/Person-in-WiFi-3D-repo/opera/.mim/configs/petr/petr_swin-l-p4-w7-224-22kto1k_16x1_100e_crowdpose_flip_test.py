_base_ = ['./petr_swin-l-p4-w7-224-22kto1k_16x1_100e_crowdpose.py']

model = dict(
    test_cfg=dict(
        max_per_img=100,
        score_thr=0.0,
        nms=dict(type='soft_nms', iou_thr=0.5)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=True,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]

data = dict(
    test=dict(pipeline=test_pipeline))
