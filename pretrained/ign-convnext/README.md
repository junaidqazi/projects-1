# IGN-ConvNeXt

ConvNeXt Tensorflow from https://github.com/bamps53/convnext-tf

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Objective

1. Provide pretrained ConvNext image segmentation for https://github.com/koechslin/swin-transformer-semantic-segmentation dataset.

And make sure put bias on the dataset, https://github.com/koechslin/swin-transformer-semantic-segmentation#information-on-the-training

```
0. padding => 1.0
1. no information (others) => 1.0
2. Dense forest => 0.5
3. Sparse forest => 1.31237
4. Moor => 1.38874
5. Herbaceous formation => 1.39761
6. Building => 1.5
7. Road => 1.47807
```

## Acknowledgement

Thanks to [KeyReply](https://www.keyreply.com/) for sponsoring GPU clouds to train the models.

## How-to

### TINY

1. Run [convert-tiny-convnext-224-to-tf1.ipynb](convert-tiny-convnext-224-to-tf1.ipynb) to convert H5 checkpoint to Tensorflow checkpoint.

2. Run [tiny.py](tiny.py) to start pretrain.

### SMALL

1. Run [convert-small-convnext-224-to-tf1.ipynb](convert-small-convnext-224-to-tf1.ipynb) to convert H5 checkpoint to Tensorflow checkpoint.

2. Run [small.py](small.py) to start pretrain.

### BASE

1. Run [convert-base-convnext-224-to-tf1.ipynb](convert-base-convnext-224-to-tf1.ipynb) to convert H5 checkpoint to Tensorflow checkpoint.

2. Run [base.py](base.py) to start pretrain.

## Download

All pretrained stored at https://huggingface.co/malay-huggingface/ign-ConvNeXt