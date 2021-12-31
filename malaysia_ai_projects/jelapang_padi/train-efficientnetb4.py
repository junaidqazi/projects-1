import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import segmentation_models as sm
import tensorflow as tf
import numpy as np
import utils_tf1
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from glob import glob
from sklearn.utils import shuffle
import random

num_train_steps = 10000
init_lr = 1e-4
end_learning_rate = 1e-6
BACKBONE = 'efficientnetb4'
BATCH_SIZE = 8
CLASSES = ['paddy']


def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]
    return x


def sometimes(aug): return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2),
    sometimes(iaa.Rot90([1, 3])),
    sometimes(iaa.Add((-2, 2))),
    sometimes(iaa.pillike.EnhanceSharpness(factor=(0, 1.5)))
], random_order=True)


def augmentation(image, mask):
    if random.random() > 0:
        segmap = SegmentationMapsOnImage(mask.astype(np.int32), shape=image.shape)
        img = image.astype(np.uint8)
        images_aug_i, segmaps_aug_i = seq(image=img, segmentation_maps=segmap)
        seg = segmaps_aug_i.draw()[0][:, :, :1]
        seg[seg > 0] = 1
        seg = seg.astype(np.float32)
        image = images_aug_i
        mask = seg
    image = preprocess_input(image)
    return image, mask


def generate():
    files = glob('*-pic/*.npy')
    while True:
        files = shuffle(files)
        for f in files:
            f_mask = f.replace('-pic/', '-mask/')
            image = (np.load(f) * 255.0).astype(np.uint8)
            mask = np.load(f_mask)
            image, mask = augmentation(image, mask)
            yield {
                'image': image,
                'mask': mask,
            }


def get_dataset(batch_size=BATCH_SIZE, shuffle_size=32, num_cpu_threads=4,
                thread_count=24):
    def get():
        d = tf.data.Dataset.from_generator(
            generate,
            {
                'image': tf.float32,
                'mask': tf.float32,
            },
            output_shapes={
                'image': tf.TensorShape([256, 256, 3]),
                'mask': tf.TensorShape([256, 256, 1]),
            },
        )
        d = d.prefetch(tf.contrib.data.AUTOTUNE)
        d = d.padded_batch(
            batch_size,
            padded_shapes={
                'image': tf.TensorShape([256, 256, 3]),
                'mask': tf.TensorShape([256, 256, 1]),
            },
            padding_values={
                'image': tf.constant(0, dtype=tf.float32),
                'mask': tf.constant(0, dtype=tf.float32),
            },
        )
        return d

    return get


def model_fn(features, labels, mode, params):
    X = features['image']
    Y = features['mask']

    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    logits = model(X)
    loss = sm.losses.binary_focal_dice_loss(Y, logits)
    iou_score = sm.metrics.IOUScore(threshold=0.5)(Y, logits)
    f_score = sm.metrics.FScore(threshold=0.5)(Y, logits)

    tf.identity(loss, 'train_loss')
    tf.identity(iou_score, 'iou_score')
    tf.identity(f_score, 'f_score')

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    init_checkpoint = 'out/model.ckpt'

    assignment_map, initialized_variable_names = utils_tf1.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if mode == tf.estimator.ModeKeys.TRAIN:

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=end_learning_rate,
            power=1.0,
            cycle=False,
        )
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss
        )

    return estimator_spec


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_loss', 'iou_score', 'f_score'], every_n_iter=1
    )
]

train_dataset = get_dataset()

utils_tf1.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='efficientnetb4-unet-jelapang-padi',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=1000,
    max_steps=num_train_steps,
    eval_fn=None,
    train_hooks=train_hooks,
)
