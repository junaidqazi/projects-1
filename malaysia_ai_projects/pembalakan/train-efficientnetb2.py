import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from efficientnet.tfkeras import EfficientNetB2
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import concatenate
import tensorflow as tf
import numpy as np
import utils_tf1
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from glob import glob
from skimage.transform import resize
import random


def sometimes(aug): return iaa.Sometimes(0.6, aug)


seq = iaa.Sequential([
    sometimes(iaa.Affine(rotate=(-90, 90))),
    iaa.Add((-5, 5)),
    sometimes(iaa.Clouds()),
    iaa.CropToFixedSize(width=256, height=256),
    sometimes(iaa.MotionBlur(k=3, angle=[-45, 45])),
], random_order=True)


def augmentation(image, ori_mask):
    mask = np.zeros_like(ori_mask)
    mask[:, :, 1] = 1.0
    mask = cv2.bitwise_and(mask, ori_mask)
    mask = mask[:, :, 1:2]
    if random.random() > 0.5:
        segmap = SegmentationMapsOnImage(mask.astype(np.int32), shape=image.shape)
        img = image.astype(np.uint8)
        images_aug_i, segmaps_aug_i = seq(image=img, segmentation_maps=segmap)
        seg = segmaps_aug_i.draw()[0][:, :, :1]
        seg[seg > 0] = 1
        seg = seg.astype(np.float32)
        images_aug_i = images_aug_i.astype(np.float32) / 255.0
        return images_aug_i, seg
    else:
        image = resize(image, (256, 256))
        mask = resize(mask, (256, 256))
        mask = np.around(mask)
        return image.astype(np.float32), mask


def _parse_image_function(example_proto):
    image_feature_description = {
        "image": tf.compat.v1.FixedLenFeature([], tf.string),
        "mask": tf.compat.v1.FixedLenFeature([], tf.string),
    }
    features = tf.compat.v1.parse_single_example(example_proto, features=image_feature_description)
    image = tf.image.decode_png(features['image'], channels=3)
    mask = tf.io.decode_raw(features['mask'], out_type="float")
    mask = tf.reshape(mask, [512, 512, 3])
    mask = tf.cast(mask, tf.float32)

    image, mask = tf.compat.v1.numpy_function(augmentation, [image, mask], [tf.float32, tf.float32])
    image = tf.reshape(image, (256, 256, 3))
    mask = tf.reshape(mask, (256, 256, 1))
    features['image'] = image
    features['mask'] = mask
    return features


def get_dataset(files, batch_size=16, shuffle_size=32, num_cpu_threads=4,
                thread_count=24, is_training=True):
    def get():
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(files))
            cycle_length = min(num_cpu_threads, len(files))
            d = d.interleave(
                tf.data.TFRecordDataset,
                cycle_length=cycle_length,
                block_length=batch_size)
            d = d.shuffle(buffer_size=50)
        else:
            d = tf.data.TFRecordDataset(files)
            d = d.repeat()
        d = d.map(_parse_image_function, num_parallel_calls=thread_count)
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


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True, training=True):
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    if activation == True:
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16, training=True):
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(blockInput)
    x = tf.keras.layers.BatchNormalization()(x)
    blockInput = tf.keras.layers.BatchNormalization()(blockInput, training=training)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = tf.math.add(x, blockInput)
    return x


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * tf.reduce_sum(intersection) + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


class Model:
    def __init__(self, X, Y, img_size=256, dropout_rate=0.25, training=True,):
        self.X = X
        self.Y = Y
        # self.X = tf.placeholder(tf.float32, (None, img_size, img_size, 3))
        # self.Y = tf.placeholder(tf.float32, (None, img_size, img_size, 1))
        backbone = EfficientNetB2(weights='imagenet',
                                  include_top=False,
                                  input_tensor=self.X)
        self.efficientnet = backbone
        start_neurons = 16

        # [B, 16, 16, 160]
        conv4 = backbone.layers[224].output
        conv4 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
        pool4 = tf.keras.layers.Dropout(dropout_rate)(pool4, training=training)
        # [B, 8, 8, 160]

        # [B, 8, 8, 512]
        convm = tf.keras.layers.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
        convm = residual_block(convm, start_neurons * 32, training=training)
        convm = residual_block(convm, start_neurons * 32, training=training)
        # [B, 8, 8, 512]
        convm = tf.keras.layers.LeakyReLU(alpha=0.1)(convm)

        # [B, 16, 16, 256]
        deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        # [B, 16, 16, 416]
        uconv4 = tf.keras.layers.Dropout(dropout_rate)(uconv4, training=training)

        # [B, 16, 16, 256]
        uconv4 = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = residual_block(uconv4, start_neurons * 16)
        uconv4 = residual_block(uconv4, start_neurons * 16)
        # [B, 16, 16, 256]
        uconv4 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv4)

        # [B, 32, 32, 128]
        deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
        # [B, 32, 32, 56]
        conv3 = backbone.layers[108].output
        # [B, 32, 32, 184]
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = tf.keras.layers.Dropout(dropout_rate)(uconv3, training=training)

        # [B, 32, 32, 128]
        uconv3 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = residual_block(uconv3, start_neurons * 8)
        uconv3 = residual_block(uconv3, start_neurons * 8)
        # [B, 32, 32, 128]
        uconv3 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv3)

        # [B, 64, 64, 64]
        deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
        # [B, 64, 64, 32]
        conv2 = backbone.layers[65].output
        # [B, 64, 64, 96]
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = tf.keras.layers.Dropout(0.1)(uconv2, training=training)

        # [B, 64, 64, 64]
        uconv2 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = residual_block(uconv2, start_neurons * 4, training=training)
        uconv2 = residual_block(uconv2, start_neurons * 4, training=training)
        uconv2 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv2)

        # [B, 128, 128, 32]
        deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
        # [B, 128, 128, 24]
        conv1 = backbone.layers[22].output
        # [B, 128, 128, 66]
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = tf.keras.layers.Dropout(0.1)(uconv1, training=training)

        # [B, 128, 128, 32]
        uconv1 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
        uconv1 = residual_block(uconv1, start_neurons * 2, training=training)
        uconv1 = residual_block(uconv1, start_neurons * 2, training=training)
        uconv1 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv1)

        # [B, 256, 256, 16]
        uconv0 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
        uconv0 = tf.keras.layers.Dropout(0.1)(uconv0, training=training)
        uconv0 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)

        uconv0 = tf.keras.layers.Dropout(dropout_rate/2)(uconv0, training=training)
        self.logits = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)


num_train_steps = 10000
init_lr = 1e-4
end_learning_rate = 1e-6


def model_fn(features, labels, mode, params):
    X = features['image']
    Y = features['mask']
    model = Model(X=X, Y=Y)
    loss = tf.reduce_mean(bce_dice_loss(model.Y, model.logits))
    tf.identity(loss, 'train_loss')

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    init_checkpoint = 'out-b2/model.ckpt'

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
        ['train_loss'], every_n_iter=1
    )
]

dataset_path = sorted(glob('content/gdrive/MyDrive/Dataset/*.tfrec'))
train_set = dataset_path[:-1]
test_set = dataset_path[-1:]

train_dataset = get_dataset(train_set, is_training=True)
test_dataset = get_dataset(test_set, is_training=False)

utils_tf1.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='efficientnetb2-unet-pembalakan',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=1000,
    max_steps=num_train_steps,
    eval_fn=test_dataset,
    train_hooks=train_hooks,
)
