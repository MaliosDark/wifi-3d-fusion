# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmcv.utils import Registry, build_from_cfg
from mmdet.core.bbox.builder import BBOX_ASSIGNERS as MMDET_BBOX_ASSIGNERS
from mmdet.core.bbox.builder import BBOX_SAMPLERS as MMDET_BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS as MMDET_BBOX_CODERS

BBOX_ASSIGNERS = Registry('bbox_assigner', parent=MMDET_BBOX_ASSIGNERS)
BBOX_SAMPLERS = Registry('bbox_sampler', parent=MMDET_BBOX_SAMPLERS)
BBOX_CODERS = Registry('bbox_coder', parent=MMDET_BBOX_CODERS)


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)
