# Installation

Opera relies on several basic packages such as MMCV, MMDetection, etc, so you need to install these packages at first.

1. Install `mmcv`

   ```bash
   cd /ROOT/Opera/third_party/mmcv
   MMCV_WITH_OPS=1 pip install -e .
   ```

2. Install `mmdet`

   ```bash
   cd /ROOT/Opera/third_party/mmdetection
   pip install -e .
   ```

3. Install `opera`

   ```bash
   cd /ROOT/Opera
   pip install -r requirements.txt
   pip install -e .
   ```

# Add new projects

If you need to add a new project in Opera, please follow:

1. Add dataset definition codes under `opera/datasets`, and use `DATASET` to register the new defined dataset class.
2. Add data pre-processing codes under `opera/datasets/pipelines`, and use `PIPELINE` to register the new defined pipeline class.
3. Add model definition codes about detectors/segmentors, backbones, neck, heads, etc, and use corresponding decorators to register them.
4. Add core modules for your projects under `opera/core`. If the added module is a anchor generator or box sampler, then you must use corresponding decorators to register it.
5. (Optional) Add compilable operators for your projects under `opera/ops`, and create corresponding Python interfaces.
6. Add unit-test cases under `tests` to cover the essential parts in your project.

# Prepare configurations

1. Opera allows users to call all the modules defined in `opera` and `third_party`.

2. You need to indicate the source of the dataset or model explicitly like

   ```python
   model1 = dict(type='mmdet.RetinaNet')
   model2 = dict(type='opera.PETR')
   data1 = dict(train=dict(type='mmdet.CocoDataset'))
   data2 = dict(train=dict(type='opera.CocoPoseDataset'))
   ```

# Unit-test

1. Install all the dependencies.

   ```bash
   pip install -r requirements/tests.txt -r requirements/optional.txt
   ```

2. Run all the unit-test cases and check the coverage.

   ```bash
   coverage run --branch --source opera -m pytest tests/
   coverage xml
   coverage report -m
   ```

# NOTES

1. Opera allows the same name from different scopes. For example, there can be a model named `ResNet` in `opera` and a model with the same name in `mmdet`. You just need to indicate the source when you want to call it in a configuration. 
2. The default scope of a registered module in a configuration is `opera`, but we strongly recommend to indicate all the modules' scope explicitly to avoid unknowable errors.
3. Please contain the abstract and copyright information at the start of your new added codes.
4. The code style should follow PEP8. 

