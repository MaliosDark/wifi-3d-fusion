# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmcv.utils import Registry, build_from_cfg
from mmdet.core.bbox.match_costs.builder import MATCH_COST as MMDET_MATCH_COS

MATCH_COST = Registry('Match Cost', parent=MMDET_MATCH_COS)


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, MATCH_COST, default_args)
