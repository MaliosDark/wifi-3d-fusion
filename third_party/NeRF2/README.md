# NeRF<sup>2</sup>: Neural Radio-Frequency Radiance Fields

Thank you for your interest in our work. This repository maintains code for NeRF<sup>2</sup>, recognized as the Best Paper Runner-Up at ACM MobiCom 2023. NeRF<sup>2</sup> is a physical-layer neural network capable of accurately predicting signal characteristics at any location based on the position of a transmitter. By integrating learned statistical models with physical ray tracing, NeRF<sup>2</sup> creates synthetic datasets ideal for training application-layer neural networks. This technology also demonstrates potential in indoor localization and 5G MIMO channel prediction, showcasing an fusion of wireless communication and AI.

![NeRF2 Example](https://github.com/XPengZhao/NeRF2/blob/gh-pages/static/images/spt-predict.jpg?raw=true)

## [Project](https://xpengzhao.github.io/NeRF2/) | [Paper](https://dl.acm.org/doi/10.1145/3570361.3592527) | [Datasets](https://1drv.ms/f/c/60d52909f2b04a6c/EtgHuC4t6xpFo1tsXzgQQfcBgoyrgZkkVwf9wcX8qxJsjQ?e=hMeEXQ)

### RFID spectrum / BLE / MIMO  prediction

Datasets and pretrained models are available at [Here](https://1drv.ms/f/c/60d52909f2b04a6c/EtgHuC4t6xpFo1tsXzgQQfcBgoyrgZkkVwf9wcX8qxJsjQ?e=hMeEXQ).

The datasets are organized as follows:

```text
NeRF2-Dataset
|-- BLE   # BLE RSSI Prediction Dataset
    |-- rssi-ckpts-1.tar         # pretrained model
    |-- rssi-dataset-1.tar.gz    # rssi dataset
|-- MIMO   # MIMO CSI Prediction Dataset
    |-- csi-ckpts-1.tar          # pretrained model
    |-- csi-dataset-1.tar.gz     # csi dataset
|-- RFID   # RFID Spectrum Prediction Dataset
    |-- s23-ckpts.tar            # pretrained model
    |-- s23-dataset.tar.gz       # spectrum dataset
```


## Running

### Spectrum prediction

**training the model**

```bash
python nerf2_runner.py --mode train --config configs/rfid-spectrum.yml --dataset_type rfid --gpu 0
```

**Inference the model**

```bash
python nerf2_runner.py --mode test --config configs/rfid-spectrum.yml --dataset_type rfid --gpu 0
```



### RSSI prediction

**training the model**

```bash
python nerf2_runner.py --mode train --config configs/ble-rssi.yml --dataset_type ble --gpu 0
```

**Inference the model**

```bash
python nerf2_runner.py --mode test --config configs/ble-rssi.yml --dataset_type ble --gpu 0
```

**MRI**

```python
python baseline/mri.py
```

### CSI prediction

**training the model**

```bash
python nerf2_runner.py --mode train --config configs/mimo-csi.yml --dataset_type mimo --gpu 0
```

**Inference the model**

```bash
python nerf2_runner.py --mode test --config configs/mimo-csi.yml --dataset_type mimo --gpu 0
```





## To-Do List

- [ ] CGAN RSSI prediction baseline
- [ ] Release more datasets
- [ ] Instruction of preparing own datasets
- [ ] Implementation on Taichi to speed up the code


Please stay tuned for updates and feel free to reach out if you have any questions or need further information.


## License

NeRF<sup>2</sup> is MIT-licensed. The license applies to the pre-trained models and datasets as well.

## Citation

If you find the repository is helpful to your project, please cite as follows:

```bibtex
@inproceedings{zhao2023nerf2,
    author = {Zhao, Xiaopeng and An, Zhenlin and Pan, Qingrui and Yang, Lei},
    title = {NeRF2: Neural Radio-Frequency Radiance Fields},
    booktitle = {Proc. of ACM MobiCom '23},
    pages = {1--15},
    year = {2023}
}
```

## Acknowledgment

Some code snippets are borrowed from [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [NeuS](https://github.com/Totoro97/NeuS).

