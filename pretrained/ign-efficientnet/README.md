# IGN-EfficientNet

EfficientNet Tensorflow from https://github.com/qubvel/efficientnet

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

## Download

All pretrained stored at https://huggingface.co/malay-huggingface/ign-efficientnet

