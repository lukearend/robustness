"""Class implementations for various datasets."""

import os
import random

import tensorflow as tf

import estimator_utils
from vgg_preprocessing import preprocess_image as vgg_preprocess_image


class ImageNetDataset(object):
    """ImageNet dataset.

    Described at http://www.image-net.org/challenges/LSVRC/2012/.
    """
    DEPTH = 3

    def __init__(self, mode, data_dir, params, predict_split='validation',
                 imagenet_train_predict_shuffle_seed=None,
                 imagenet_train_predict_partial=False):
        self.mode = mode
        self.data_dir = data_dir
        self.params = params
        self.predict_split = predict_split
        self.imagenet_train_predict_shuffle_seed = imagenet_train_predict_shuffle_seed
        self.imagenet_train_predict_partial = imagenet_train_predict_partial

    def get_filenames(self):
        """Get names of data files."""
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            lookup_name = 'train'
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            lookup_name = 'validation'
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            lookup_name = self.predict_split
        filenames = tf.gfile.Glob(
            os.path.join(self.data_dir, '{}-*-of-*'.format(lookup_name)))
        if tf.estimator.ModeKeys.PREDICT:
            # Sort so that TFRecords will be read out deterministically.
            filenames = sorted(filenames)
        return filenames

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label Tensors."""
        features = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/synset': tf.FixedLenFeature([], tf.string),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(serialized_example, features)

        # Get label as a Tensor.
        label = parsed_features['image/class/label']

        # Decode the image JPEG string into a Tensor.
        image = tf.image.decode_jpeg(parsed_features['image/encoded'],
                                     channels=self.DEPTH)

        # VGG preprocessing borrowed from slim; includes data augmentation so train_with_distortion should be set to True.
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            assert self.params['train_with_distortion'] == True
            is_training = True
        else:
            is_training = False
        image = vgg_preprocess_image(image, 224, 224, is_training=is_training)

        return image, label

    def make_batch(self, batch_size):
        """Make a batch of images and labels."""
        filenames = self.get_filenames()
        if self.mode == tf.estimator.ModeKeys.PREDICT and self.imagenet_train_predict_partial:
            # Sort and shuffle with seed to randomize deterministically.
            random.seed(self.imagenet_train_predict_shuffle_seed)
            random.shuffle(filenames)
        dataset = tf.contrib.data.TFRecordDataset(filenames)

        # Parse records.
        dataset = dataset.map(self.parser,
                              num_threads=batch_size,
                              output_buffer_size=2 * batch_size)

        # If training, shuffle and repeat indefinitely.
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=50000 + 3 * batch_size)
            dataset = dataset.repeat(-1)
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            if self.predict_split == 'train':
                if self.imagenet_train_predict_partial:
                    MAX_EXAMPLES = 50000
                    # Skip to start at a random spot in the first TFRecord.
                    random.seed(self.imagenet_train_predict_shuffle_seed)
                    skip_examples = random.randint(0, 1251)
                    dataset = dataset.skip(skip_examples)
                    # Continue shuffling amongst at least as many examples
                    # as it could see in 3 cross validations.
                    dataset.shuffle(buffer_size=3 * MAX_EXAMPLES,
                                    seed=self.imagenet_train_predict_shuffle_seed)
                    num_examples = MAX_EXAMPLES
                else:
                    # Take whole training set.
                    num_examples = self.num_examples_per_epoch(tf.estimator.ModeKeys.TRAIN)
            else:
                # Take whole validation set.
                num_examples = self.num_examples_per_epoch(tf.estimator.ModeKeys.EVAL)
            # Take as much of the dataset as possible that can be evenly
            # divided by batch_size.
            while True:
                if num_examples % batch_size == 0:
                        break
                else:
                    num_examples -= 1
            dataset = dataset.take(num_examples)
            dataset = dataset.repeat(1)

            # dataset = dataset.take(1000) # For fast debugging!
        else:
            dataset = dataset.repeat(1)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    @staticmethod
    def num_examples_per_epoch(mode):
        """Stores constants about this dataset."""
        if mode == tf.estimator.ModeKeys.TRAIN:
            return 1281167
        return 50000


class Cifar10Dataset(object):
    """CIFAR-10 dataset.

    Described at http://www.cs.toronto.edu/~kriz/cifar.html.
    """
    HEIGHT = 32
    WIDTH = 32
    DEPTH = 3

    def __init__(self, mode, data_dir, params, predict_split='validation'):
        self.mode = mode
        self.data_dir = data_dir
        self.params = params
        self.predict_split = predict_split

    def get_filenames(self):
        """Get names of data files."""
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            lookup_name = 'train'
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            lookup_name = 'validation'
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            lookup_name = self.predict_split
        filenames = tf.gfile.Glob(
            os.path.join(self.data_dir, '{}-*-of-*'.format(lookup_name)))
        if tf.estimator.ModeKeys.PREDICT:
            # Sort so that TFRecords will be read out deterministically.
            filenames = sorted(filenames)
        return filenames

    def parser(self, serialized_example):
        """Parse a single tf.Example into image and label Tensors."""
        features = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], tf.string),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/fixation_pt': tf.FixedLenFeature([2], tf.float32)}
        parsed_features = tf.parse_single_example(serialized_example, features)

        # Get label as a Tensor.
        label = parsed_features['image/class/label']

        # Decode the image JPEG string into a Tensor.
        image = tf.image.decode_jpeg(parsed_features['image/encoded'],
                                     channels=self.DEPTH)

        # Convert from uint8 -> float32 and map onto range [0, 1].
        image = tf.cast(image, tf.float32) * (1. / 255)

        # Standardize image.
        image = tf.image.per_image_standardization(image)

        # Apply data augmentation.
        if (self.mode == tf.estimator.ModeKeys.TRAIN
            and self.params['train_with_distortion']):
            # Randomly flip the image, zero-pad with four pixels along
            # each edge, and take a random 32 x 32 crop.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.image.crop_to_bounding_box(image,
                tf.random_uniform([], minval=0, maxval=8, dtype=tf.int32),
                tf.random_uniform([], minval=0, maxval=8, dtype=tf.int32),
                32, 32)

        return image, label

    def make_batch(self, batch_size):
        """Make a batch of images and labels."""
        filenames = self.get_filenames()
        dataset = tf.contrib.data.TFRecordDataset(filenames)

        # Parse records.
        dataset = dataset.map(self.parser,
                              num_threads=batch_size,
                              output_buffer_size=2 * batch_size)

        # If training, shuffle and repeat indefinitely.
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10000 + 3 * batch_size)
            dataset = dataset.repeat(-1)
        elif self.mode == tf.estimator.ModeKeys.PREDICT:
            if self.predict_split == 'train':
                num_examples = self.num_examples_per_epoch(tf.estimator.ModeKeys.TRAIN)
            else:
                num_examples = self.num_examples_per_epoch(tf.estimator.ModeKeys.EVAL)
            # Take as much of the dataset as possible that can be evenly
            # divided by batch_size.
            while True:
                if num_examples % batch_size == 0:
                    break
                else:
                    num_examples -= 1
            dataset = dataset.take(num_examples)
            dataset = dataset.repeat(1)

            # dataset = dataset.take(1000) # For fast debugging!
        else:
            dataset = dataset.repeat(1)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    @staticmethod
    def num_examples_per_epoch(mode):
        """Stores constants about this dataset."""
        if mode == tf.estimator.ModeKeys.TRAIN:
            return 45000
        return 5000
