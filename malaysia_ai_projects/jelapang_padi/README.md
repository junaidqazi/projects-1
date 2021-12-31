# Jelapang Padi

Satellite Semantic segmentation for paddy field in Malaysia.

## Dataset

All dataset can get at https://huggingface.co/datasets/malay-huggingface/jelapang-padi

1. [mapbox-kedah.ipynb](mapbox-kedah.ipynb) to scrap satellite images for specific latlong boundaries at Kedah with 15 zoom level.
2. [mapbox-sekinchan.ipynb](mapbox-sekinchan.ipynb) to scrap satellite images for specific latlong boundaries at Sekinchan with 15 zoom level.

## Checkpoint

All checkpoints can get at https://huggingface.co/malay-huggingface/jelapang-padi

## how-to train

1. Prepare dataset, [prepare-dataset.ipynb](prepare-dataset.ipynb).

### EfficientNet B4

1. Convert Keras checkpoint to TF1 checkpoint, [convert-efficientnetb4-keras-to-tf1.ipynb](convert-efficientnetb4-keras-to-tf1.ipynb).

2. Run training script,

```bash
python3 train-efficientnetb4.py
```

### EfficientNet B2

1. Convert Keras checkpoint to TF1 checkpoint, [convert-efficientnetb2-keras-to-tf1.ipynb](convert-efficientnetb2-keras-to-tf1.ipynb).

2. Run training script,

```bash
python3 train-efficientnetb2.py
```