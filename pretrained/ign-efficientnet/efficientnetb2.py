import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from efficientnet.tfkeras import EfficientNetB2
from tensorflow.keras.layers import concatenate
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from glob import glob
import tensorflow as tf
import numpy as np
import utils_tf1
import imgaug as ia
import imgaug.augmenters as iaa
from skimage.transform import resize
import segmentation_models as sm
from sklearn.utils import shuffle
import random

IMAGE_SIZE = 384
CLASSES = 7
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

        # [B, IMAGE_SIZE, IMAGE_SIZE, 16]
        uconv0 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
        uconv0 = tf.keras.layers.Dropout(0.1)(uconv0, training=training)
        uconv0 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = residual_block(uconv0, start_neurons * 1, training=training)
        uconv0 = tf.keras.layers.LeakyReLU(alpha=0.1)(uconv0)

        uconv0 = tf.keras.layers.Dropout(dropout_rate/2)(uconv0, training=training)
        self.logits = tf.keras.layers.Conv2D(CLASSES, (1, 1), padding="same", activation="softmax")(uconv0)


num_train_steps = 10000
init_lr = 1e-3
end_learning_rate = 1e-5


def model_fn(features, labels, mode, params):
    X = features['image']
    Y = features['mask']
    Y = tf.one_hot(Y, depth=CLASSES)
    model = Model(X=X, Y=Y)

    focal_loss = sm.losses.CategoricalFocalLoss()
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.5, 1.31237, 1.38874, 1.39761, 1.5, 1.47807, 1.0]))
    total_loss = dice_loss + (1 * focal_loss)

    iou = sm.metrics.IOUScore(threshold=0.5)
    fscore = sm.metrics.FScore(threshold=0.5)

    loss = total_loss(model.Y, model.logits)
    tf.identity(loss, 'loss')

    iou_score = iou(model.Y, model.logits)
    tf.identity(iou_score, 'iou')
    tf.summary.scalar('iou', iou)

    fscore_score = fscore(model.Y, model.logits)
    tf.identity(fscore_score, 'fscore')
    tf.summary.scalar('fscore', fscore)

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
        ['loss', 'iou', 'fscore'], every_n_iter=1
    )
]

train_dataset = get_dataset('train')
test_dataset = get_dataset('test')

utils_tf1.run_training(
    train_fn=train_dataset,
    model_fn=model_fn,
    model_dir='efficientnetb2-ign',
    num_gpus=1,
    log_step=1,
    save_checkpoint_step=1000,
    max_steps=num_train_steps,
    eval_fn=test_dataset,
    train_hooks=train_hooks,
)
