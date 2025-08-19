# dataset settings
dataset_type = 'opera.WifiPoseDataset'
data_root = '/home/qianbo/wifipose/data/'
train_pipeline = [
    dict(type='opera.DefaultFormatBundle',
         extra_keys=['gt_keypoints', 'gt_labels']),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'],
         meta_keys=[]),
]

test_pipeline = [
    dict(
        type='mmdet.MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='opera.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_labels']),
            dict(type='mmdet.Collect',
                keys=['img'],
                meta_keys=[]),
        ])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
# test_pipeline = [
#     dict(type='mmdet.LoadImageFromFile'),
#     dict(
#         type='mmdet.MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='mmdet.Resize', keep_ratio=True),
#             dict(type='mmdet.RandomFlip'),
#             dict(type='mmdet.Normalize', **img_norm_cfg),
#             dict(type='mmdet.Pad', size_divisor=1),
#             dict(type='mmdet.ImageToTensor', keys=['img']),
#             dict(type='mmdet.Collect', keys=['img']),
#         ])
# ]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_root=data_root+'train_data',
        pipeline=train_pipeline,
        mode='train'),
    val=dict(
        type=dataset_type,
        dataset_root=data_root+'test_data',
        pipeline=test_pipeline,
        mode='test'),
    test=dict(
        type=dataset_type,
        dataset_root=data_root+'test_data',
        pipeline=test_pipeline,
        mode='test'))
evaluation = dict(interval=1, metric='mpjpe')
