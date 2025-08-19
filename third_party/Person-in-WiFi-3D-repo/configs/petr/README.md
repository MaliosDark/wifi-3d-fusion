# [End-to-End Multi-Person Pose Estimation with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Shi_End-to-End_Multi-Person_Pose_Estimation_With_Transformers_CVPR_2022_paper.pdf)

## Results and Models

### COCO

| Model | Backbone | Lr schd | mAP  | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | Config | Download |
|:-----:|:--------:|:-------:|:----:|:---------------:|:---------------:|:--------------:|:--------------:|:------:|:--------:|
| PETR  |  R-50    |  100e   | 68.8 |      87.5       |      76.3       |      62.7      |      77.7      | [config](https://github.com/hikvision-research/opera/blob/main/configs/petr/petr_r50_16x2_100e_coco.py) | [Google Drive](https://drive.google.com/file/d/1HcwraqWdZ3CaGMQOJHY8exNem7UnFkfS/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1C0HbQWV7K-GHQE7q34nUZw?pwd=u798) |
| PETR  |  R-101   |  100e   | 70.0 |      88.5       |      77.5       |      63.6      |      79.4      | [config](https://github.com/hikvision-research/opera/blob/main/configs/petr/petr_r101_16x2_100e_coco.py) | [Google Drive](https://drive.google.com/file/d/1O261Jrt4JRGlIKTmLtPy3AUruwX1hsDf/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1D5wqNP53KNOKKE5NnO2Dnw?pwd=keyn) |
| PETR  |  Swin-L  |  100e   | 73.1 |      90.7       |      80.9       |      67.2      |      81.7      | [config](https://github.com/hikvision-research/opera/blob/main/configs/petr/petr_swin-l-p4-w7-224-22kto1k_16x1_100e_coco.py) | [Google Drive](https://drive.google.com/file/d/1ujL0Gm5tPjweT0-gdDGkTc7xXrEt6gBP/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1X5Cdq75GosRCKqbHZTSpJQ?pwd=t9ea) |

### CrowdPose

| Model | Backbone | Lr schd | Flip test | mAP  | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>E</sup> | AP<sup>M</sup> | AP<sup>H</sup> | Config | Download |
|:-----:|:--------:|:-------:|:---------:|:----:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|:------:|:--------:|
| PETR  |  Swin-L  |  100e   |     N     | 71.7 |      90.0       |      78.3       |      77.5      |      72.0      |      65.8      | [config](https://github.com/hikvision-research/opera/blob/main/configs/petr/petr_swin-l-p4-w7-224-22kto1k_16x1_100e_crowdpose.py) | [Google Drive](https://drive.google.com/file/d/1aS-TIFuSC2gVfmr5n4qrtmQSLFTbP6Lm/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/17shA0nSJO3PJJPLwD-YoHQ?pwd=qr3g) |
| PETR  |  Swin-L  |  100e   |     Y     | 72.3 |      90.8       |      78.8       |      78.7      |      72.9      |      65.5      | [config](https://github.com/hikvision-research/opera/blob/main/configs/petr/petr_swin-l-p4-w7-224-22kto1k_16x1_100e_crowdpose_flip_test.py) | [Google Drive](https://drive.google.com/file/d/1aS-TIFuSC2gVfmr5n4qrtmQSLFTbP6Lm/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/17shA0nSJO3PJJPLwD-YoHQ?pwd=qr3g) |
## NOTE

1. Swin-L are trained with batch size 16 due to GPU memory limitation.
2. The performance is unstable. `PETR` may fluctuate about 0.2 mAP.

## Citation

```BibTeX
@inproceedings{shi2022end,
  title={End-to-End Multi-Person Pose Estimation With Transformers},
  author={Shi, Dahu and Wei, Xing and Li, Liangqi and Ren, Ye and Tan, Wenming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11069--11078},
  year={2022}
}
```
