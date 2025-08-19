# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formatting import to_tensor
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle \
    as MMDetDefaultFormatBundle

from ..builder import PIPELINES


@PIPELINES.register_module()
class DefaultFormatBundle(MMDetDefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    Besides, it is extended to support other customed fields, such as
    "gt_keypoints", "gt_areas", etc.
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)
    - customed_field1: (1)to tensor, (2)to DataContainer
    - customed_field2: (1)to tensor, (2)to DataContainer
    """

    def __init__(self,
                 *args,
                 extra_keys=[],
                 **kwargs):
        super(DefaultFormatBundle, self).__init__(*args, **kwargs)
        self.extra_keys = extra_keys

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        # results = super(DefaultFormatBundle, self).__call__(results)
        # assert isinstance(self.extra_keys, (list, tuple))
        if self.extra_keys:
            for key in self.extra_keys:
                if key not in results:
                    continue
                results[key] = DC(to_tensor(results[key]))
        return results
