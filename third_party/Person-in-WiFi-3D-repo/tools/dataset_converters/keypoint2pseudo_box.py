# Copyright (c) Hikvision Research Institute. All rights reserved.
import json
import copy
import numpy as np


def get_pseudo_bbox(kpts):
    tmp_kpts = []
    for kpt in kpts:
        if kpt[2] > 0:
            tmp_kpts.append(kpt)
    num_kpt = len(tmp_kpts)
    if num_kpt == 0:
        return None
    tmp_kpts = np.array(tmp_kpts).reshape([num_kpt, 3])
    x = tmp_kpts[:, 0]
    y = tmp_kpts[:, 1]
    x1 = x.min()
    y1 = y.min()
    x2 = x.max()
    y2 = y.max()
    w = x2 - x1
    h = y2 - y1
    bbox = np.array([x1, y1, w, h])
    bbox = bbox.tolist()
    return bbox


def transform(data):
    new_data = dict()
    new_data['info'] = data['info']
    new_data['licenses'] = data['licenses']
    new_data['images'] = data['images']
    new_data['categories'] = data['categories']
    anns = data['annotations']
    new_anns = copy.deepcopy(anns)
    for ann in new_anns:
        kpts = np.array(ann['keypoints'], dtype=np.float32).reshape([17, 3])
        new_bbox = get_pseudo_bbox(kpts)
        if new_bbox is not None:
            ann['bbox'] = new_bbox

    new_data['annotations'] = new_anns
    return new_data


def main():
    cur_path = '/dataset/public/coco/annotations/'
    cur_file = cur_path + 'person_keypoints_train2017.json'
    new_path = '/data/configs/inspose/'
    new_file = new_path + 'person_keypoints_train2017_pseudobox.json'
    with open(cur_file, 'r') as f:
        data = json.load(f)
    new_data = transform(data)
    with open(new_file, 'w') as g:
        json.dump(new_data, g)


if __name__ == '__main__':
    main()
