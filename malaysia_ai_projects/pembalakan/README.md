# Pembalakan

Satellite Semantic segmentation for deforestation in Malaysia.

## Dataset

All dataset can get at https://github.com/BioWar/Satellite-Image-Segmentation-using-Deep-Learning-for-Deforestation-Detection/tree/main/Dataset

## Checkpoint

All checkpoints can get at https://huggingface.co/malay-huggingface/pembalakan

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