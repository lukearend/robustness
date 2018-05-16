"""Class implementations for various datasets."""

import os

import tensorflow as tf

import estimator_utils


class ImageNetDataset(object):
    """ImageNet dataset.

    Described at http://www.image-net.org/challenges/LSVRC/2012/.
    """
    HEIGHT = 256
    WIDTH = 256
    DEPTH = 3

    def __init__(self, mode, data_dir, params, predict_split='validation',
                 imagenet_train_predict_shuffle_seed=None,
                 imagenet_train_predict_just_some=False):
        self.mode = mode
        self.data_dir = data_dir
        self.params = params
        self.predict_split = predict_split
        self.imagenet_train_predict_shuffle_seed = imagenet_train_predict_shuffle_seed

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
        if self.mode == tf.estimator.ModeKeys.PREDICT and self.imagenet_train_predict_shuffle_seed is not None:
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

        # Convert from uint8 -> float32 and map onto range [0, 1].
        image = tf.cast(image, tf.float32) * (1. / 255.)

        # Subtract mean ImageNet activation.
        mean_imagenet_rgb = estimator_utils.get_mean_imagenet_rgb()
        image = image - mean_imagenet_rgb

        # Apply data augmentation.
        if (self.mode == tf.estimator.ModeKeys.TRAIN and
            self.params['train_with_distortion']):
            # Randomly flip the image, and then randomly crop from
            # the central 224 x 224.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.crop_to_bounding_box(image,
                tf.random_uniform([], minval=0, maxval=32, dtype=tf.int32),
                tf.random_uniform([], minval=0, maxval=32, dtype=tf.int32),
                224, 224)
        else:
            # Take a central 224 x 224 crop.
            image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

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
            dataset = dataset.shuffle(buffer_size=50000 + 3 * batch_size)
            dataset = dataset.repeat(-1)
        elif self.mode == tf.estimator.ModeKeys.PREDICT and self.predict_split == 'train':
            if self.imagenet_train_predict_just_some:
                # Shuffle the whole ImageNet in memory and take 50000.
                # Requires several hundred GB of memory, but the easiest way
                # to do this a more extensive implementation.
                dataset = dataset.shuffle(buffer_size=1281167,
                                          seed=self.imagenet_train_predict_shuffle_seed)
                    # IMPORTANT (and sketchy): assume batch size 100
                    # so this is divided evenly by the batches.
                    dataset = dataset.take(50000)
            else:
                # IMPORTANT (and sketchy): assume batch size 100
                # so this is divided evenly by the batches.
                dataset = dataset.take(1281100)
            dataset = dataset.repeat(1)
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
