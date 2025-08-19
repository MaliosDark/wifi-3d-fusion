# Copyright (c) Hikvision Research Institute. All rights reserved.
from mmdet.core import bbox2result
from mmdet.models.detectors.detr import DETR
from mmdet.models.detectors.single_stage import SingleStageDetector

from ..builder import DETECTORS


@DETECTORS.register_module()
class SOIT(DETR):
    """Implementation of `SOIT: Segmenting Objects with 
    Instance-Aware Transformers <https://arxiv.org/abs/2112.11037>`."""

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_masks, 
                                              gt_bboxes_ignore)
        return losses
        
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            img (list[torch.Tensor]): List of multiple images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray, np.ndarray]]: BBox and mask results 
                of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_masks in bbox_list
        ]
        mask_results = [res[2] for res in bbox_list]
        results = [(bbox_results, mask_results)
            for bbox_results, mask_results in zip(bbox_results, mask_results)]

        return results
