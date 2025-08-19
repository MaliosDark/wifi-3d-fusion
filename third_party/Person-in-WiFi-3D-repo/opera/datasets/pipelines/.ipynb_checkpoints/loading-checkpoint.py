# Copyright (c) Hikvision Research Institute. All rights reserved.
import os

import numpy as np
from mmdet.datasets.pipelines import LoadAnnotations as MMDetLoadAnnotations

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotations(MMDetLoadAnnotations):
    """Load multiple types of annotations.

    Args:
        with_keypoint (bool): Whether to parse and load the keypoint annotation.
            Default: False.
        with_area (bool): Whether to parse and load the mask area annotation.
            Default: False.
    """

    def __init__(self,
                 *args,
                 with_keypoint=False,
                 with_area=False,
                 **kwargs):
        super(LoadAnnotations, self).__init__(*args, **kwargs)
        self.with_keypoint = with_keypoint
        self.with_area = with_area

    def _load_keypoints(self, results):
        """Private function to load keypoint annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded keypoint annotations.
        """

        ann_info = results['ann_info']
        results['gt_keypoints'] = ann_info['keypoints'].copy()
        results['keypoint_fields'].append('gt_keypoints')
        return results

    def _load_areas(self, results):
        """Private function to load mask area annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask area annotations.
        """

        ann_info = results['ann_info']
        results['gt_areas'] = ann_info['areas'].copy()
        results['area_fields'].append('gt_areas')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask,
                semantic segmentation, keypoint and mask area annotations.
        """

        results = super(LoadAnnotations, self).__call__(results)
        if results is None:
            return None
        if self.with_keypoint:
            results = self._load_keypoints(results)
        if self.with_area:
            results = self._load_areas(results)
        return results

    def __repr__(self):
        repr_str = super(LoadAnnotations, self).__repr__()[:-1] + ', '
        repr_str += f'with_keypoint={self.with_keypoint}, '
        repr_str += f'with_area={self.with_area})'
        return repr_str
