# Copyright (c) Hikvision Research Institute. All rights reserved.
import torch
import numpy as np


def distance2keypoint(points, offset, max_shape=None):
    """Decode distance prediction to keypiont.

    Args:
        points (Tensor): Shape (N, 2).
        offset (Tensor): Offset from the given point to K keypoints (N, K*2).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded keypoints.
    """
    if offset.size(0) == 0:
        return offset.new_zeros((0, offset.shape[1] // 2, 2))
    
    offset = offset.reshape(offset.shape[0], -1, 2)
    points = points[:, None, :].expand(offset.shape)
    keypoints = points + offset
    if max_shape is not None:
        keypoints[:, :, 0].clamp_(min=0, max=max_shape[1])
        keypoints[:, :, 1].clamp_(min=0, max=max_shape[0])
    
    return keypoints


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(1, 2, 0).contiguous()
    feat = feat.view(-1, feat.size(2))
    dim = feat.size(1)
    ind = ind.unsqueeze(1).expand(ind.size(0), dim)
    feat = feat.gather(0, ind)
    return feat


def gaussian_radius(det_size, min_overlap=0.7):
    """calculate gaussian radius according to object size.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y = torch.arange(-m, m + 1, dtype=torch.float32, device=m.device)[:, None]
    x = torch.arange(-n, n + 1, dtype=torch.float32, device=m.device)[None, :]
    # y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(np.float32).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])
    radius = int(radius)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]
    # assert masked_gaussian.eq(1).float().sum() == 1
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        heatmap[y - top:y + bottom, x - left:x + right] = torch.max(
            masked_heatmap, masked_gaussian * k)
    return heatmap


def draw_short_range_offset(offset_map, mask_map, gt_kp, radius):
    gt_kp_int = torch.floor(gt_kp)
    x_coord = gt_kp[0] - \
        (torch.arange(-radius, radius + 1, dtype=torch.float32,
                      device=offset_map.device) + gt_kp_int[0])
    y_coord = gt_kp[1] - \
        (torch.arange(-radius, radius + 1, dtype=torch.float32,
                      device=offset_map.device) + gt_kp_int[1])
    y_map, x_map = torch.meshgrid(y_coord, x_coord)
    short_offset = torch.cat([x_map.unsqueeze(0), y_map.unsqueeze(0)], dim=0)

    x, y = int(gt_kp_int[0]), int(gt_kp_int[1])
    radius = int(radius)

    height, width = offset_map.shape[1:]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_offset_map = offset_map[:, y - top:y + bottom, x - left:x + right]
    masked_short_offset = short_offset[:, radius - top:radius + bottom,
                                       radius - left:radius + right]
    if min(masked_short_offset.shape) > 0 and min(masked_offset_map.shape) > 0:
        offset_map_distance = torch.pow(masked_offset_map, 2).sum(
            dim=0, keepdim=True).expand(masked_offset_map.shape)
        short_offset_distance = torch.pow(masked_short_offset, 2).sum(
            dim=0, keepdim=True).expand(masked_short_offset.shape)
        offset_map[:, y - top:y + bottom, x - left:x + right] = torch.where(
            short_offset_distance < offset_map_distance,
            masked_short_offset, masked_offset_map)
        mask_map[:, y - top:y + bottom, x - left:x + right] = 1
    return offset_map, mask_map


def bbox_kpt2result(bboxes, labels, kpts, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5).
        labels (torch.Tensor | np.ndarray): shape (n, ).
        kpts (torch.Tensor | np.ndarray): shape (n, K, 3).
        num_classes (int): class number, including background class.

    Returns:
        list(ndarray): bbox and keypoint results of each class.
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)], \
            [np.zeros((0, kpts.size(1), 3), dtype=np.float32)
                for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            kpts = kpts.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)], \
            [kpts[labels == i, :, :] for i in range(num_classes)]


def kpt_flip(kpts, img_shape, flip_pairs, direction):
    """Flip keypoints horizontally.

    Args:
        kpts (Tensor): Shape (n, K, 2).
        img_shape (tuple): Image shape.
        flip_pairs (list): Flip pair index.
        direction (str): Flip direction, only "horizontal" is supported now.
            Default: "horizontal".

    Returns:
        Tensor: Flipped keypoints.
    """
    assert kpts.shape[-1] % 2 == 0
    assert direction == 'horizontal'
    flipped = kpts.clone()
    flipped[..., 0] = img_shape[1] - flipped[..., 0]
    for pair in flip_pairs:
        flipped[:, pair, :] = flipped[:, pair[::-1], :]
    return flipped


def kpt_mapping_back(kpts, img_shape, scale_factor, flip, flip_direction):
    """Map keypoints from testing scale to original image scale."""
    if kpts.shape[1] == 17:
        from opera.datasets import CocoPoseDataset
        flip_pairs = CocoPoseDataset.FLIP_PAIRS
    elif kpts.shape[1] == 14:
        from opera.datasets import CrowdPoseDataset
        flip_pairs = CrowdPoseDataset.FLIP_PAIRS
    else:
        raise NotImplementedError
    new_kpts = kpt_flip(kpts, img_shape, flip_pairs, flip_direction) \
        if flip else kpts
    new_kpts = new_kpts.view(-1, 2) / new_kpts.new_tensor(scale_factor[:2])
    return new_kpts.view(kpts.shape)
