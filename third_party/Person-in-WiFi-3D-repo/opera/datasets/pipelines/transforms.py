# Copyright (c) Hikvision Research Institute. All rights reserved.
import math

import cv2
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.pipelines import Resize as MMDetResize
from mmdet.datasets.pipelines import RandomFlip as MMDetRandomFlip
from mmdet.datasets.pipelines import RandomCrop as MMDetRandomCrop

from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize(MMDetResize):
    """Resize images & bbox & mask & keypoint & mask area.

    Args:
        keypoint_clip_border (bool, optional): Whether to clip the objects
            outside the border of the image. Defaults to True.
    """

    def __init__(self,
                 *args,
                 keypoint_clip_border=True,
                 **kwargs):
        super(Resize, self).__init__(*args, **kwargs)
        self.keypoint_clip_border = keypoint_clip_border

    def _resize_keypoints(self, results):
        """Resize keypoints with ``results['scale_factor']``."""
        for key in results.get('keypoint_fields', []):
            keypoints = results[key].copy()
            keypoints[:,
                      0::3] = keypoints[:, 0::3] * results['scale_factor'][0]
            keypoints[:,
                      1::3] = keypoints[:, 1::3] * results['scale_factor'][1]
            if self.keypoint_clip_border:
                img_shape = results['img_shape']
                keypoints[:, 0::3] = np.clip(keypoints[:, 0::3], 0,
                                             img_shape[1])
                keypoints[:, 1::3] = np.clip(keypoints[:, 1::3], 0,
                                             img_shape[0])
            results[key] = keypoints

    def _resize_areas(self, results):
        """Resize mask areas with ``results['scale_factor']``."""
        for key in results.get('area_fields', []):
            areas = results[key].copy()
            areas = areas * results['scale_factor'][0] * results[
                'scale_factor'][1]
            results[key] = areas

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map, keypoints, mask areas.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        results = super(Resize, self).__call__(results)
        self._resize_keypoints(results)
        self._resize_areas(results)
        return results

    def __repr__(self):
        repr_str = super(Resize, self).__repr__()[:-1] + ', '
        repr_str += f'keypoint_clip_border={self.keypoint_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip(MMDetRandomFlip):
    """Flip the image & bbox & mask & keypoint.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.
    """

    def keypoint_flip(self, keypoints, img_shape, direction, flip_pairs):
        """Flip keypoints horizontally.

        Args:
            keypoints (numpy.ndarray): person's keypoints, shape (..., K*3).
            img_shape (tuple[int]): Image shape (height, width).
            direction (str): Flip direction. Only 'horizontal' is supported.
            flip_pairs (list): Flip pair indices.

        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        assert keypoints.shape[-1] % 3 == 0
        flipped = keypoints.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped = flipped.reshape(flipped.shape[0], flipped.shape[1] // 3,
                                      3)
            valid_idx = flipped[..., -1] > 0
            flipped[valid_idx, 0] = w - flipped[valid_idx, 0]
            for pair in flip_pairs:
                flipped[:, pair, :] = flipped[:, pair[::-1], :]
            flipped[..., 0] = np.clip(flipped[..., 0], 0, w)
            flipped = flipped.reshape(flipped.shape[0], keypoints.shape[1])
        elif direction == 'vertical':
            raise NotImplementedError
        elif direction == 'diagonal':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps, keypoints.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        results = super(RandomFlip, self).__call__(results)
        if results['flip']:
            # flip keypoints
            for key in results.get('keypoint_fields', []):
                results[key] = self.keypoint_flip(
                    results[key], results['img_shape'],
                    results['flip_direction'],
                    results['ann_info']['flip_pairs'])
        return results


@PIPELINES.register_module()
class RandomCrop(MMDetRandomCrop):
    """Random crop the image & bboxes & masks & keypoints & mask areas.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        kpt_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - The keys for bboxes, keypoints and areas must be aligned. That is,
          `gt_bboxes` corresponds to `gt_keypoints` and `gt_areas`, and
          `gt_bboxes_ignore` corresponds to `gt_keypoints_ignore` and
          `gt_areas_ignore`.
    """

    def __init__(self,
                 *args,
                 kpt_clip_border=True,
                 **kwargs):
        super(RandomCrop, self).__init__(*args, **kwargs)
        self.kpt_clip_border = kpt_clip_border
        # The key correspondence from bboxes to kpts and areas.
        self.bbox2kpt = {
            'gt_bboxes': 'gt_keypoints',
            'gt_bboxes_ignore': 'gt_keypoints_ignore'
        }
        self.bbox2area = {
            'gt_bboxes': 'gt_areas',
            'gt_bboxes_ignore': 'gt_areas_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps, keypoints, mask areas.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

            # keypoint fields, e.g. gt_keypoints
            kpt_key = self.bbox2kpt.get(key)
            if kpt_key in results:
                results[kpt_key] = results[kpt_key][valid_inds]
            
            # mask area fields, e.g. gt_areas
            area_key = self.bbox2area.get(key)
            if area_key in results:
                results[area_key] = results[area_key][valid_inds]

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # crop keypoints accordingly and clip to the image boundary
        for key in results.get('keypoint_fields', []):
            # e.g. gt_keypoints
            if len(results[key]) > 0:
                kpt_offset = np.array([offset_w, offset_h], dtype=np.float32)
                keypoints = results[key].copy()
                keypoints = keypoints.reshape(keypoints.shape[0], -1, 3)
                keypoints[..., :2] = keypoints[..., :2] - kpt_offset
                invalid_idx = \
                    (keypoints[..., 0] < 0).astype(np.int8) | \
                    (keypoints[..., 1] < 0).astype(np.int8) | \
                    (keypoints[..., 0] > img_shape[1]).astype(np.int8) | \
                    (keypoints[..., 1] > img_shape[0]).astype(np.int8) | \
                    (keypoints[..., 2] < 0.1).astype(np.int8)
                assert key == 'gt_keypoints'
                gt_valid = ~invalid_idx.all(1)
                results['gt_bboxes'] = results['gt_bboxes'][gt_valid]
                results['gt_areas'] = results['gt_areas'][gt_valid]
                results['gt_labels'] = results['gt_labels'][gt_valid]
                keypoints[invalid_idx > 0, :] = 0
                keypoints = keypoints[gt_valid]
                if len(keypoints) == 0:
                    return None
                keypoints = keypoints.reshape(keypoints.shape[0], -1)
                if self.kpt_clip_border:
                    keypoints[:, 0::3] = np.clip(keypoints[:, 0::3], 0,
                                                 img_shape[1])
                    keypoints[:, 1::3] = np.clip(keypoints[:, 1::3], 0,
                                                 img_shape[0])
                results[key] = keypoints

        # assert len(results['gt_bboxes']) == len(results['gt_keypoints'])
        return results

    def __repr__(self):
        repr_str = super(RandomCrop, self).__repr__()[:-1] + ', '
        repr_str += f'kpt_clip_border={self.kpt_clip_border})'
        return repr_str


@PIPELINES.register_module()
class KeypointRandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
    """

    def __init__(self,
                 max_rotate_degree=10.0,
                 max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(114, 114, 114),
                 min_bbox_size=2,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Center
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        center_matrix[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
            @ center_matrix)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        bboxes = results['gt_bboxes']
        keypoints = results['gt_keypoints']
        assert len(bboxes) == len(keypoints)
        num_bboxes = len(bboxes)
        if num_bboxes:
            # homogeneous coordinates
            xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
            ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
            ones = np.ones_like(xs)
            points = np.vstack([xs, ys, ones])

            warp_points = warp_matrix @ points
            warp_points = warp_points[:2] / warp_points[2]
            xs = warp_points[0].reshape(num_bboxes, 4)
            ys = warp_points[1].reshape(num_bboxes, 4)

            warp_bboxes = np.vstack(
                (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

            warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
            warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

            # keypoints
            kps = keypoints.reshape(num_bboxes, -1, 3)
            kpxs = kps[:, :, 0].reshape(-1)
            kpys = kps[:, :, 1].reshape(-1)
            kp_vis = kps[:, :, 2]
            kpones = np.ones_like(kpxs)
            kps = np.vstack([kpxs, kpys, kpones])
            warp_kps = warp_matrix @ kps
            warp_kps = warp_kps[:2] / warp_kps[2]
            kpxs = warp_kps[0].reshape(num_bboxes, -1, 1)
            kpys = warp_kps[1].reshape(num_bboxes, -1, 1)
            warp_kps = np.concatenate((kpxs, kpys, kp_vis[..., None]), axis=-1)

            # filter keypoints
            kp_invalid, gt_valid = self.filter_gt_keypoints(
                warp_kps, (height, width))
            warp_kps[kp_invalid > 0, :] = 0
            warp_kps = warp_kps[gt_valid]
            if len(warp_kps) == 0:
                return None

            results['gt_bboxes'] = warp_bboxes[gt_valid]
            # TODO: change areas after affine for detection task
            results['gt_areas'] = results['gt_areas'][gt_valid]
            results['gt_labels'] = results['gt_labels'][gt_valid]
            results['gt_keypoints'] = warp_kps.reshape(sum(gt_valid), -1)

            # # filter bboxes
            # valid_index = self.filter_gt_bboxes(bboxes * scaling_ratio,
            #                                     warp_bboxes)
            # results['gt_bboxes'] = warp_bboxes[valid_index]
            # results['gt_keypoints'] = warp_kps[valid_index]
            # if 'gt_labels' in results:
            #     results['gt_labels'] = results['gt_labels'][
            #         valid_index]
            # if 'gt_masks' in results:
            #     raise NotImplementedError(
            #         'RandomAffine only supports bbox.')
        return results

    def filter_gt_keypoints(self, keypoints, img_shape):
        invalid_idx = (keypoints[..., 0] < 0).astype(np.int8) | \
                      (keypoints[..., 1] < 0).astype(np.int8) | \
                      (keypoints[..., 0] > img_shape[1]).astype(np.int8) | \
                      (keypoints[..., 1] > img_shape[0]).astype(np.int8) | \
                      (keypoints[..., 2] < 0.1).astype(np.int8)
        gt_valid = ~invalid_idx.all(1)
        return invalid_idx, gt_valid

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix
