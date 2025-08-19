# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn.bricks.transformer import ATTENTION as MMCV_ATTENTION
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING as \
    MMCV_POSITIONAL_ENCODING
from mmcv.cnn.bricks.transformer import TRANSFORMER_LAYER_SEQUENCE as \
    MMCV_TRANSFORMER_LAYER_SEQUENCE
from mmdet.models.utils.builder import TRANSFORMER as MMDET_TRANSFORMER


ATTENTION = Registry('attention', parent=MMCV_ATTENTION)
POSITIONAL_ENCODING = Registry('Position encoding',
                               parent=MMCV_POSITIONAL_ENCODING)
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence',
                                      parent=MMCV_TRANSFORMER_LAYER_SEQUENCE)
TRANSFORMER = Registry('Transformer', parent=MMDET_TRANSFORMER)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)
