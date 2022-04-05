import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from convnext import ConvNeXt, model_configs
from tensorflow.keras.layers import concatenate
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from glob import glob
import tensorflow as tf
import numpy as np
import utils_tf1
import imgaug as ia
import imgaug.augmenters as iaa
import segmentation_models as sm
from skimage.transform import resize
from sklearn.utils import shuffle
import random
import adamw

IMAGE_SIZE = 384
CLASSES = 8
BATCH_SIZE = 8


def sometimes(aug): return iaa.Sometimes(0.6, aug)


seq = iaa.Sequential([
    sometimes(iaa.Affine(rotate=(-90, 90))),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Add((-5, 5)),
    sometimes(iaa.Clouds()),
    iaa.CropToFixedSize(width=IMAGE_SIZE, height=IMAGE_SIZE),
    sometimes(iaa.MotionBlur(k=3, angle=[-45, 45])),
], random_order=True)


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


def augmentation(image, mask):
    if random.random() > 0.5:
        segmap = SegmentationMapsOnImage(mask.astype(np.int32), shape=image.shape)
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        mask = segmaps_aug_i.get_arr()
        image = images_aug_i
    else:
        image = (resize(image, (IMAGE_SIZE, IMAGE_SIZE)) * 255)
        mask = (resize(mask, (IMAGE_SIZE, IMAGE_SIZE)) * 255)
    image = preprocess_input(image)
    return image.astype('float32'), mask.astype('int32')


def generate(prefix):
    prefix = prefix.decode()
    files = glob(f'{prefix}_image/*.npy')
    while True:
        files = shuffle(files)
        for f in files:
            f_mask = f.replace('_image/', '_mask/')
            image = np.load(f)
            mask = np.load(f_mask)
            image, mask = augmentation(image, mask)
            yield {
                'image': image,
                'mask': mask,
            }


def get_dataset(prefix, batch_size=BATCH_SIZE, shuffle_size=32, num_cpu_threads=4,
                thread_count=24):
    def get():
        d = tf.data.Dataset.from_generator(
            generate,
            {
                'image': tf.float32,
                'mask': tf.int32,
            },
            output_shapes={
                'image': tf.TensorShape([IMAGE_SIZE, IMAGE_SIZE, 3]),
                'mask': tf.TensorShape([IMAGE_SIZE, IMAGE_SIZE]),
            },
            args=(prefix,),
        )
        d = d.prefetch(tf.contrib.data.AUTOTUNE)
        d = d.padded_batch(
            batch_size,
            padded_shapes={
                'image': tf.TensorShape([IMAGE_SIZE, IMAGE_SIZE, 3]),
                'mask': tf.TensorShape([IMAGE_SIZE, IMAGE_SIZE]),
            },
            padding_values={
                'image': tf.constant(0, dtype=tf.float32),
                'mask': tf.constant(0, dtype=tf.int32),
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


class Model:
    def __init__(self, X, Y, img_size=IMAGE_SIZE, dropout_rate=0.25, training=True,):

        self.X = X
        self.Y = Y

        img_size = 384
        num_classes = 1000
        include_top = False
        cfg = model_configs['convnext_base']
        net = ConvNeXt(num_classes, cfg['depths'],
                       cfg['dims'], include_top)
        tf.keras.layers.Input(tensor=self.X)
        out = net(self.X)
        backbone = tf.keras.Model(self.X, out)
        start_neurons = 16

        # [B, 12, 12, 768]
        conv4 = backbone.layers[1].layers[3].layers[1].output
        conv4 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
        pool4 = tf.keras.layers.Dropout(dropout_rate)(pool4, training=training)
        # [B, 6, 6, 768]

        # [B, 6, 6, 512]
        convm = tf.keras.layers.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
        convm = residual_block(convm, start_neurons * 32, training=training)
        convm = residual_block(convm, start_neurons * 32, training=training)
        # [B, 6, 6, 512]
        convm = tf.keras.layers.LeakyReLU(alpha=0.1)(convm)

        # [B, 12, 12, 256]
        deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        # [B, 12, 12, 1024]
        uconv4 = tf.keras.layers.Dropout(dropout_rate)(uconv4, training=training)

        # [B, 12, 12, 256]
        uconv4 = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
        uconv4 = residual_block(uconv4, start_neurons * 16)
        uconv4 = residual_block(uconv4, start_neurons * 16)
        # [B, 12, 12, 256]
        uconv4 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv4)

        # [B, 24, 24, 128]
        deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
        # [B, 24, 24, 384]
        conv3 = backbone.layers[1].layers[2].layers[1].output
        # [B, 24, 24, 512]
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = tf.keras.layers.Dropout(dropout_rate)(uconv3, training=training)

        # [B, 24, 24, 128]
        uconv3 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
        uconv3 = residual_block(uconv3, start_neurons * 8)
        uconv3 = residual_block(uconv3, start_neurons * 8)
        # [B, 24, 24, 128]
        uconv3 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv3)

        # [B, 48, 48, 64]
        deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
        # [B, 48, 48, 192]
        conv2 = backbone.layers[1].layers[1].layers[1].output
        # [B, 48, 48, 256]
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = tf.keras.layers.Dropout(0.1)(uconv2, training=training)

        # [B, 48, 48, 64]
        uconv2 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
        uconv2 = residual_block(uconv2, start_neurons * 4, training=training)
        uconv2 = residual_block(uconv2, start_neurons * 4, training=training)
        uconv2 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv2)

        # [B, 96, 96, 32]
        deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
        # [B, 96, 96, 96]
        conv1 = backbone.layers[1].layers[0].layers[1].output
        # [B, 96, 96, 128]
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = tf.keras.layers.Dropout(0.1)(uconv1, training=training)

        # [B, 192, 192, 16]
        uconv0 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
        uconv0 = tf.keras.layers.Dropout(0.1)(uconv0, training=training)
        uconv0 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)

        # [B, 384, 384, 16]
        uconv0 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv0)
        uconv0 = tf.keras.layers.Dropout(0.1)(uconv0, training=training)
        uconv0 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)

        uconv0 = tf.keras.layers.Dropout(dropout_rate/2)(uconv0, training=training)
        self.logits = tf.keras.layers.Conv2D(CLASSES, (1, 1), padding="same", activation="softmax")(uconv0)


num_train_steps = 30000


def model_fn(features, labels, mode, params):
    X = features['image']
    Y = features['mask']
    Y = tf.one_hot(Y, depth=CLASSES)
    model = Model(X=X, Y=Y)

    focal_loss = sm.losses.CategoricalFocalLoss()
    dice_loss = sm.losses.DiceLoss()
    total_loss = dice_loss + (1 * focal_loss)

    iou = sm.metrics.IOUScore(threshold=0.5)
    fscore = sm.metrics.FScore(threshold=0.5)

    loss = total_loss(model.Y, model.logits)
    tf.identity(loss, 'loss')

    iou_score = iou(model.Y, model.logits)
    tf.identity(iou_score, 'iou')
    tf.summary.scalar('iou', iou_score)

    fscore_score = fscore(model.Y, model.logits)
    tf.identity(fscore_score, 'fscore')
    tf.summary.scalar('fscore', fscore_score)

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    init_checkpoint = 'convnext_base_22k_1k_224/model.ckpt'

    assignment_map, initialized_variable_names = utils_tf1.get_assignment_map_from_checkpoint(
        variables, init_checkpoint
    )
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if mode == tf.estimator.ModeKeys.TRAIN:

        train_op = adamw.create_optimizer(
            loss,
            init_lr=0.0001,
            num_train_steps=num_train_steps,
            num_warmup_steps=int(0.02 * num_train_steps),
            end_learning_rate=0.00001,
            weight_decay_rate=0.001,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-6,
            clip_norm=1.0,
        )
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
        ['loss', 'iou', 'fscore'], every_n_iter=1
    )
]

train_dataset = get_dataset('train')
test_dataset = get_dataset('test')

utils_tf1.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='base-convnext-ign',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=2500,
    max_steps=num_train_steps,
    eval_fn=test_dataset,
    train_hooks=train_hooks,
)
