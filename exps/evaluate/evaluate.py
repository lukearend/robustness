"""Evaluate a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import argparse

import itertools
import math

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from tensorflow.tensorboard.backend.event_processing import event_accumulator


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--scale_factor', type=float, required=True)
parser.add_argument('--disable_batch_norm', dest='use_batch_norm',
                    action='store_false')
parser.add_argument('--dataset', type=str, required=True)
FLAGS = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


#######################################################################
# resnet_model
#######################################################################
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 1
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.contrib.layers.batch_norm(
        inputs=inputs,
        data_format='NCHW' if data_format == 'channels_first' else 'NHWC',
        decay=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, is_training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                    Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        data_format=data_format)

def _building_block_v1(inputs, filters, use_batch_norm, training,
                       projection_shortcut, strides, data_format):
    """A single block for ResNet v1, without a bottleneck.

    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        use_batch_norm: Whether to use batch normalization.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        if use_batch_norm:
            shortcut = batch_norm(inputs=shortcut, training=training,
                            data_format=data_format)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)
    if use_batch_norm:
        inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)
    if use_batch_norm:
        inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs

def block_layer(inputs, filters, use_batch_norm, bottleneck, block_fn, blocks,
                strides, training, name, data_format):
    """Creates one layer of blocks for the ResNet model.
    
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first convolution of the layer.
        use_batch_norm: Whether to use batch normalization.
        bottleneck: Is the block created a bottleneck block.
        block_fn: The block to use within the model, either `building_block` or
            `bottleneck_block`.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
        name: A string name for the tensor output of the block layer.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, use_batch_norm, training,
                    projection_shortcut, strides, data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, use_batch_norm, training, None, 1,
                      data_format)

    return tf.identity(inputs, name)


class Model(object):
    """Base class for building the Resnet Model."""
    
    def __init__(self, resnet_size, use_batch_norm, bottleneck, num_classes,
               num_filters, kernel_size, conv_stride, first_pool_size,
               first_pool_stride, block_sizes, block_strides, final_size,
               resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE):
        """Creates a model for classifying an image.

        Args:
            resnet_size: A single integer for the size of the ResNet model.
            use_batch_norm: Whether to use batch normalization.
            bottleneck: Use regular blocks or bottleneck blocks.
            num_classes: The number of classes used as labels.
            num_filters: The number of filters to use for the first block layer
                of the model. This number is then doubled for each subsequent block
                layer.
            kernel_size: The kernel size to use for convolution.
            conv_stride: stride size for the initial convolutional layer
            first_pool_size: Pool size to be used for the first pooling layer.
                If none, the first pooling layer is skipped.
            first_pool_stride: stride size for the first pooling layer. Not used
                if first_pool_size is None.
            block_sizes: A list containing n values, where n is the number of sets of
                block layers desired. Each value should be the number of blocks in the
                i-th set.
            block_strides: List of integers representing the desired stride size for
                each of the sets of block layers. Should be same length as block_sizes.
            final_size: The expected size of the model after the second pooling.
            resnet_version: Integer representing which version of the ResNet network
                to use. See README for details. Valid values: [1, 2]
            data_format: Input format ('channels_last', 'channels_first', or None).
                If set to None, the format is dependent on whether a GPU is available.
            dtype: The TensorFlow dtype to use for calculations. If not specified
                tf.float32 is used.

        Raises:
            ValueError: if invalid version is selected.
        """
        self.resnet_size = resnet_size

        if not data_format:
            data_format = (
                'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2. See README for citations.')

        self.use_batch_norm = use_batch_norm
        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if resnet_version == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size
        self.dtype = dtype

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.

        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to change
        the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would have
        been called if no custom getter was used. Custom getters typically get a
        variable with `getter`, then modify it in some way.

        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        Args:
            getter: The underlying variable getter, that has the same signature as
                tf.get_variable and returns a variable.
            name: The name of the variable to get.
            shape: The shape of the variable to get.
            dtype: The dtype of the variable to get. Note that if this is a low
                precision dtype, the variable will be created as a tf.float32 variable,
                then cast to the appropriate dtype
            *args: Additional arguments to pass unmodified to getter.
            **kwargs: Additional keyword arguments to pass unmodified to getter.

        Returns:
            A variable which is cast to fp16 if necessary.
        """

        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Returns a variable scope that the model should be created under.

        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.

        Returns:
            A variable scope for the model.
        """

        return tf.variable_scope('resnet_model',
                                 custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.

        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = block_layer(
                    inputs=inputs, filters=num_filters,
                    use_batch_norm=self.use_batch_norm, bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], training=training,
                    name='block_layer{}'.format(i + 1), data_format=self.data_format)

            if self.use_batch_norm:
                inputs = batch_norm(inputs, training, self.data_format)
            inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axis=axes, keep_dims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')
        
            inputs = tf.reshape(inputs, [-1, self.final_size])
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs


#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################


#######################################################################
# estimator_utils
#######################################################################
def get_mean_imagenet_rgb():
    """Fetches the mean activation per pixel over the entire ImageNet
    dataset from mean_imagenet_rgb.mat file, which must be located in
    the path shown."""
    # Load the data from file.
    if os.path.exists('/raid/poggio/home/larend/robust/prep/mean_imagenet_rgb.mat'):
        data = scipy.io.loadmat('/raid/poggio/home/larend/robust/prep/mean_imagenet_rgb.mat')
    elif os.path.exists('/cbcl/cbcl01/larend/robust/prep/mean_imagenet_rgb.mat'):
        data = scipy.io.loadmat('/cbcl/cbcl01/larend/robust/prep/mean_imagenet_rgb.mat')
    else:
        data = scipy.io.loadmat('/om/user/larend/robust/prep/mean_imagenet_rgb.mat')
    mean_imagenet_rgb = data['mean_imagenet_rgb']

    # Convert to float32 Tensor and map onto range [0, 1].
    mean_imagenet_rgb = tf.cast(mean_imagenet_rgb, tf.float32) * (1. / 255.)

    return mean_imagenet_rgb

def get_dataset(dataset, mode, input_path, params):
    """Get a dataset based on name."""
    if dataset == 'mnist':
        return MNISTDataset(mode,
                                            input_path,
                                            params)
    elif dataset == 'cifar10':
        return Cifar10Dataset(mode,
                                              input_path,
                                              params)
    elif dataset == 'imagenet':
        return ImageNetDataset(mode,
                                               input_path,
                                               params)
    else:
        raise ValueError("Dataset '{}' not recognized.".format(dataset))

def get_num_examples_per_epoch(dataset, mode):
    """Get number of examples per epoch for some dataset and mode."""
    if dataset == 'mnist':
        return MNISTDataset.num_examples_per_epoch(mode)
    elif dataset == 'cifar10':
        return Cifar10Dataset.num_examples_per_epoch(mode)
    elif dataset == 'imagenet':
        return ImageNetDataset.num_examples_per_epoch(mode)
    else:
        raise ValueError("Dataset '{}' not recognized.".format(dataset))

def local_device_setter(ps_device_type='cpu', worker_device='/cpu:0',
                        ps_strategy=None):
    """Set local device."""
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(1)

    def _local_device_chooser(op):
        """Choose local device."""
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        worker_device_spec = pydev.DeviceSpec.from_string(
            worker_device or "")
        worker_device_spec.merge_from(current_device)
        return worker_device_spec.to_string()
    return _local_device_chooser

def in_top_k(predictions, targets, k):
    """My implementation of tf.nn.in_top_k() which breaks ties by
    choosing the lowest index.

    Args:
        predictions: a [batch_size, classes] Tensor of probabilities.
        targets: a [batch_size] Tensor of class labels.
        k: number of top elements to look at for computing precision.

    Returns:
        correct: a [batch_size] float Tensor where 1.0 is correct and
                 0.0 is incorrect (float makes it simpler to average).
    """
    _, top_k_indices = tf.nn.top_k(predictions, k)

    # Initialize a False [batch_size] bool Tensor.
    last_correct = tf.logical_and(False, tf.is_finite(tf.to_float(targets)))
    # Loop over the top k predictions, checking if any are correct.
    for i in range(k):
        p_i = tf.squeeze(tf.slice(top_k_indices, [0, i], [-1, 1]), axis=1)
        correct = tf.logical_or(last_correct, tf.equal(tf.to_int64(targets),
                                                       tf.to_int64(p_i)))
        last_correct = correct

    return tf.to_float(correct)

class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """Hook to print out examples per second. Tracks total time and
    divides by the total number of steps to get average step time.
    batch_size is then used to determine the running average of
    examples per second. Also logs examples per second for the most
    recent interval.
    """
    def __init__(self, batch_size, every_n_steps=100, every_n_secs=None):
        """Initializer for ExamplesPerSecondHook.
            Args:
                batch_size: Total batch size used to calculate
                            examples/second from global time.
                every_n_steps: Log stats every n steps.
                every_n_secs: Log stats every n seconds.
        """
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError("Exactly one of every_n_steps and every_n_secs"
                             "should be provided.")
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._global_step_tensor = None
        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            (elapsed_time,
             elapsed_steps) = self._timer.update_last_triggered_step(
                                  global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = (self._batch_size *
                                            self._total_steps /
                                            self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                logging.info(
                    "Average examples/sec: {:.2f} ({:.2f}), step = {}".format(
                        average_examples_per_sec,
                        current_examples_per_sec,
                        self._total_steps))

def replace_monitors_with_hooks(monitors_and_hooks, model):
    """Converts a list which may contain hooks or monitors to a list of
    hooks."""
    return monitor_lib.replace_monitors_with_hooks(monitors_and_hooks, model)


#######################################################################
# estimator_datasets
#######################################################################
class ImageNetDataset(object):
    """ImageNet dataset.

    Described at http://www.image-net.org/challenges/LSVRC/2012/.
    """
    HEIGHT = 256
    WIDTH = 256
    DEPTH = 3

    def __init__(self, mode, data_dir, params):
        self.mode = mode
        self.data_dir = data_dir
        self.params = params

        if self.mode not in [tf.estimator.ModeKeys.TRAIN,
                             tf.estimator.ModeKeys.EVAL]:
            raise ValueError("Invalid mode: '{}'.".format(self.mode))

    def get_filenames(self):
        """Get names of data files."""
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            lookup_name = 'train'
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            lookup_name = 'validation'
        filenames = tf.gfile.Glob(
            os.path.join(self.data_dir, '{}-*-of-*'.format(lookup_name)))
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
        mean_imagenet_rgb = get_mean_imagenet_rgb()
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

    def __init__(self, mode, data_dir, params):
        self.mode = mode
        self.data_dir = data_dir
        self.params = params

        if self.mode not in [tf.estimator.ModeKeys.TRAIN,
                             tf.estimator.ModeKeys.EVAL]:
            raise ValueError("Invalid mode: '{}'.".format(self.mode))

    def get_filenames(self):
        """Get names of data files."""
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            lookup_name = 'train'
        elif self.mode == tf.estimator.ModeKeys.EVAL:
            lookup_name = 'validation'
        filenames = tf.gfile.Glob(
            os.path.join(self.data_dir, '{}-*-of-*'.format(lookup_name)))
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
            image = tf.image.random_flip_left_right(image)

        # Resize to 224 x 224.
        image = tf.image.resize_images(image, (224, 224))

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


#######################################################################
# estimator_graph
#######################################################################
def forward_pass(x, is_training, params):
    resnet = Model(
        resnet_size=18,
        use_batch_norm=params['use_batch_norm'],
        bottleneck=False,
        num_classes={
            'imagenet': 1000,
            'cifar10': 10}[params['dataset']],
        num_filters=params['num_filters'],
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=[2, 2, 2, 2],
        block_strides=[2, 2, 2, 2],
        final_size=params['num_filters'] * 8)

    y = resnet(x, is_training)

    return y


#######################################################################
# estimator_fns
#######################################################################
def input_fn(mode, input_path, params, num_gpus=None):
    """Create input graph for model.

    Args:
        mode: one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
        input_path: directory where input data is located.
        params: hyperparameters for training and architecture.
        num_gpus: number of GPUs participating in data-parallelism.

    Returns:
        feature_shards: list of length num_gpus containing feature
                        batches.
        label_shards: list of length num_gpus containing label batches.
    """
    with tf.device('/cpu:0'):
        if mode == tf.estimator.ModeKeys.PREDICT:
            dataset = RawImageDataset(mode, input_path, params)
            image_batch = dataset.make_batch(params['batch_size'])
            return image_batch, None

        elif mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            # Set the batch size.
            batch_size = params['batch_size']

            if mode == tf.estimator.ModeKeys.EVAL:
                # The validation batch size must evenly divide the
                # validation set to avoid the risk of getting a batch
                # with fewer than [the number of GPU] examples. Here we
                # find the largest factor of the validation set size
                # which is smaller than params['batch_size'] to use as
                # the batch size during validation. Avoid preparing a
                # validation data set with a prime number of examples.
                num_validation_examples = (
                    get_num_examples_per_epoch(
                        params['dataset'], tf.estimator.ModeKeys.EVAL))
                while True:
                    if num_validation_examples % batch_size == 0:
                        break
                    batch_size = batch_size - 1

            dataset = get_dataset(params['dataset'],
                                               mode,
                                               input_path,
                                               params)

            image_batch, label_batch = dataset.make_batch(batch_size)

        if num_gpus <= 1:
            return [image_batch], [label_batch]

        # Shard-up the batch among GPUs. Assumes image_batch will always
        # have batch_size elements since batch_size either divides the
        # data set size perfectly or the data repeats indefinitely.
        image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_gpus)]
        label_shards = [[] for i in range(num_gpus)]

        for i in range(batch_size):
            idx = i % num_gpus
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.stack(x) for x in feature_shards]
        label_shards = [tf.stack(x) for x in label_shards]

        return feature_shards, label_shards


def get_model_fn(num_gpus, variable_strategy='GPU', keep_checkpoint_max=10,
                 keep_checkpoint_every_n_hours=2):
    """Returns a function that will build the estimator.

    Args:
        num_gpus: number of GPUs to use for training.
        variable_strategy: one of {'CPU', 'GPU'}.
                           If 'CPU', CPU is the parameter server and
                           manages gradient updates.
                           If 'GPU', parameters are distributed evenly
                           across all GPUs and the first GPU manages
                           gradient updates. This is optimal with
                           interconnected K80s.
        keep_checkpoint_max: number of recent checkpoint to keep during
                             training.
        keep_checkpoint_every_n_hours: how frequently to keep a
                                       checkpoint permanently.

    Returns:
        The estimator model_fn.
    """

    def _estimator_model_fn(features, labels, mode, params):
        """Model body.

        Args:
            features: list of Tensors, one for each tower. If mode is
                      PREDICT, Tensor.
            labels: list of Tensors, one for each tower. If mode is
                    PREDICT, None.
            mode: one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}.
            params: hyperparameters for training and architecture.
        Returns:
            EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if mode == tf.estimator.ModeKeys.PREDICT:
            logits = forward_pass(features,
                                                  is_training,
                                                  params)

            predictions = {
                'classes': tf.argmax(logits, axis=1),
                'probabilities': tf.nn.softmax(logits),
            }

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        elif mode in [tf.estimator.ModeKeys.TRAIN,
                      tf.estimator.ModeKeys.EVAL]:
            tower_features = features
            tower_labels = labels
            tower_losses = []
            tower_gradvars = []
            tower_predictions = []
            weight_loss = None

            if num_gpus == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = num_gpus
                device_type = 'gpu'

            for i in range(num_devices):
                worker_device = '/{}:{}'.format(device_type, i)
                if variable_strategy == 'CPU':
                    device_setter = local_device_setter(
                        worker_device=worker_device)
                elif variable_strategy == 'GPU':
                    device_setter = local_device_setter(
                        ps_device_type='gpu',
                        worker_device=worker_device,
                        ps_strategy=\
                            tf.contrib.training.GreedyLoadBalancingStrategy(
                                num_gpus,
                                tf.contrib.training.byte_size_load_fn))

                # Compute loss and gradients for each tower.
                with tf.variable_scope('estimator', reuse=bool(i != 0)):
                    with tf.name_scope('tower_{}'.format(i)):
                        with tf.device(device_setter):
                            (loss,
                             gradvars,
                             predictions,
                             weight_loss) = _tower_fn(tower_features[i],
                                                      tower_labels[i],
                                                      is_training,
                                                      params)

                            tower_losses.append(loss)
                            tower_gradvars.append(gradvars)
                            tower_predictions.append(predictions)

            # Compute global loss and gradients.
            gradvars = []
            with tf.name_scope('gradient_averaging'):
                all_grads = {}
                for grad, var in itertools.chain(*tower_gradvars):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in all_grads.items():
                    # Average gradients on device holding their
                    # variables.
                    with tf.device(var.device):
                        if len(grads) == 1:
                            avg_grads = grads[0]
                        else:
                            avg_grads = tf.multiply(tf.add_n(grads),
                                                    1. / len(grads))
                    gradvars.append((avg_grads, var))

            # Set device that runs ops to apply global gradient updates.
            if variable_strategy == 'GPU':
                consolidation_device = '/gpu:0'
            else:
                consolidation_device = '/cpu:0'
            with tf.device(consolidation_device):
                # Calculate the learning rate decay schedule.
                global_step = tf.train.get_or_create_global_step()
                num_examples_per_epoch = (
                    get_num_examples_per_epoch(
                        params['dataset'], tf.estimator.ModeKeys.TRAIN))
                num_steps_per_decay = int(num_examples_per_epoch /
                                          params['batch_size'] *
                                          params['num_epochs_per_decay'])
                learning_rate = tf.train.exponential_decay(
                    tf.to_float(params['initial_learning_rate']),
                    global_step,
                    num_steps_per_decay,
                    params['learning_rate_decay_factor'],
                    staircase=True,
                    name='learning_rate')

                # Create momentum gradient descent optimizer.
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    params['momentum'])

                # Create the train op.
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(gradvars,
                                                         global_step=global_step)

                # Combine results and compute metrics; name tensors we
                # want to log.
                loss = tf.reduce_mean(tower_losses, name='loss')
                weight_loss = tf.identity(weight_loss, name='weight_loss')
                cross_entropy_loss = tf.subtract(loss,
                                                 weight_loss,
                                                 name='cross_entropy_loss')

                predictions = {
                    'classes': tf.concat(
                        [p['classes'] for p in tower_predictions],
                        axis=0),
                    'probabilities': tf.concat(
                        [p['probabilities'] for p in tower_predictions],
                        axis=0)}
                stacked_labels = tf.concat(labels, axis=0)

                in_top_one = in_top_k(
                    predictions['probabilities'],
                    stacked_labels,
                    1)
                in_top_five = in_top_k(
                    predictions['probabilities'],
                    stacked_labels,
                    5)

                precision_at_one = tf.reduce_mean(in_top_one,
                                                  name='precision_at_one')
                precision_at_five = tf.reduce_mean(in_top_five,
                                                   name='precision_at_five')

                eval_metric_ops = {
                    'loss/loss': tf.metrics.mean(loss),
                    'precision/precision_at_one':
                        tf.metrics.mean(in_top_one),
                    'precision/precision_at_five':
                        tf.metrics.mean(in_top_five)}

            # Write summaries to monitor the training process.
            if is_training:
                # Plot helpful training metrics.
                tf.summary.scalar('global_step/learning_rate',
                                  learning_rate)
                tf.summary.scalar('loss/loss', loss)
                tf.summary.scalar('loss/weight_loss', weight_loss)
                tf.summary.scalar('loss/cross_entropy_loss',
                                  cross_entropy_loss)
                tf.summary.scalar('precision/precision_at_one',
                                  precision_at_one)
                tf.summary.scalar('precision/precision_at_five',
                                  precision_at_five)

                # Visualize a few training examples.
                tf.summary.image('examples/images',
                                 tower_features[0])

            # Merge the summaries for saving.
            summary_op = tf.summary.merge_all()

            # Create a saver for saving to checkpoints.
            saver = tf.train.Saver(
                max_to_keep=keep_checkpoint_max,
                keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

            scaffold = tf.train.Scaffold(summary_op=summary_op,
                                         saver=saver)

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=eval_metric_ops,
                                              scaffold=scaffold)

    return _estimator_model_fn

def _tower_fn(features, labels, is_training, params):
    """Build computation tower.

    Args:
        features: a batch of images.
        labels: a batch of labels.
        is_training: bool, true if training graph.
        params: hyperparameters specifying architecture of the model.

    Returns:
        loss: loss for the tower.
        gradvars: a tuple containing the gradients and parameters.
        predictions: unscaled logit predictions.
        weight_loss: total loss from L2 penalty on model weights.
    """
    logits = forward_pass(features,
                                          is_training,
                                          params)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    cross_entropy_loss = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
    model_params = tf.trainable_variables()
    weight_loss = (params['weight_decay'] *
                   tf.add_n([tf.nn.l2_loss(v) for v in model_params]))
    loss = cross_entropy_loss + weight_loss

    grads = tf.gradients(loss, model_params)
    gradvars = zip(grads, model_params)

    return loss, gradvars, predictions, weight_loss


#######################################################################
# estimator
#######################################################################
class Estimator(object):
    """A class that wraps up the estimator."""
    DEFAULT_PARAMS = {
        'initial_learning_rate': 0.1,
        'learning_rate_decay_factor': 0.1,
        'num_epochs_per_decay': 30,
        'max_epochs': 120,
        'train_with_distortion': True,
        'momentum': 0.9,
        'batch_size': 256,
        'weight_decay': 0.0001,
        'dataset': 'imagenet',
        'use_batch_norm': True,
        'num_filters': 64
    }

    def __init__(self, model_dir='/tmp', params={}, tf_random_seed=12435):
        self.model_dir = model_dir
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(params)
        self.tf_random_seed = tf_random_seed

        self.session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(
                # Use only as much GPU memory as needed based on
                # runtime allocations.
                allow_growth=True,
                # Force all CPU tensors to be allocated with Cuda
                # pinned memory. May hurt host performance if model is
                # extremely large, as all Cuda pinned memory is
                # unpageable.
                force_gpu_compatible=True))

    def evaluate(self, data_dir='/tmp', num_gpus=1):
        """Evaluate the model.

        Args:
            data_dir: directory in which to look for validation data.
            num_gpus: number of GPUs to use.
        """
        # Configure and build the model.
        model_fn = get_model_fn(num_gpus)

        config = tf.estimator.RunConfig().replace(
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=config,
                                       params=self.params)

        # Configure evaluation.
        eval_input_fn = lambda: input_fn(tf.estimator.ModeKeys.EVAL,
                                               data_dir,
                                               self.params,
                                               num_gpus)

        # Evaluate the model.
        model.evaluate(input_fn=eval_input_fn)


#######################################################################
# run evaluation
#######################################################################
def main():
    if dataset == 'cifar10':
    num_filters = {
        0.25: 16,
        0.5: 32,
        1: 64,
        2: 128,
        4: 256}[FLAGS.scale_factor]

    data_dir = {
        'cifar10': '/om/user/larend/data/cifar-10-tfrecords',
        'imagenet': '/om/user/larend/data/imagenet-tfrecords'}[FLAGS.dataset]

    model = Estimator(
        model_dir=FLAGS.model_dir,
        params={
            'batch_size': 100,
            'dataset': FLAGS.dataset,
            'use_batch_norm': FLAGS.use_batch_norm,
            'num_filters': {
                0.25: 16,
                0.5: 32,
                1: 64,
                2: 128,
                4: 256}[FLAGS.scale_factor]},
        tf_random_seed=12345)

    model.evaluate(
        data_dir={
            'cifar10': '/om/user/larend/data/cifar-10-tfrecords',
            'imagenet': '/om/user/larend/data/imagenet-tfrecords'}[FLAGS.dataset])


if __name__ == '__main__':
    main()
