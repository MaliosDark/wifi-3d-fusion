# [SOIT: Segmenting Objects with Instance-Aware Transformers](https://arxiv.org/abs/2112.11037)

## Introduction

A fully end-to-end instance segmentation method based on Transformers.

## Results and Models
| Backbone  | Lr schd | Dataset | Test scale | mAP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | mAP<sup>box</sup> | Config | Download |
|:---------:|:-------:|:-------:|:----------:|:----:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|:-----------------:|:------:|:--------:|
| R-50      | 50e     |  COCO   |(1333, 800) | 42.2 |      64.6       |      45.3       |      23.1      |      45.3      |      61.8      |       48.9        | [config](https://github.com/hikvision-research/opera/blob/main/configs/soit/soit_r50_16x2_50e_coco.py) | [Google Drive](https://drive.google.com/file/d/1-Eu7BkmmrU4gLK4fw8gqTs7II-96RA6x/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1XFZ1rR5wT_h3f0BMmCvn0Q?pwd=fgfv) |
| R-101     | 50e     |  COCO   |(1333, 800) | 42.9 |      65.7       |      46.0       |      23.1      |      46.4      |      63.3      |       49.5        | [config](https://github.com/hikvision-research/opera/blob/main/configs/soit/soit_r101_16x2_50e_coco.py) | [Google Drive](https://drive.google.com/file/d/1xU1i4bYV-HoiH5ctpPSA7ky4vdSlH-_r/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1sw0vqGzhHAl8ZUAfnLeHCA?pwd=kava) |

## NOTE

1. AP without superscript denotes mask AP. mAP<sup>box</sup> denotes bbox AP.

## Citation

```BibTeX
@inproceedings{yu2022soit,
  title={SOIT: Segmenting Objects with Instance-Aware Transformers},
  author={Yu, Xiaodong and Shi, Dahu and Wei, Xing and Ren, Ye and Ye, Tingqun and Tan, Wenming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
