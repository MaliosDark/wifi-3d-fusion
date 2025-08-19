_base_ = './petr_r50_16x2_100e_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
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
        in_channels=[384, 768, 1536]))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
# optimizer
optimizer = dict(lr=1e-4)
