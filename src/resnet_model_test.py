# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

import perturbations as pt

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
                       projection_shortcut, strides, data_format,
                       perturbation_type, perturbation_amount, kill_mask):
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

        kill_mask: here is a 2-element list with the kill masks for both
            convolutions in the block.

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

    # Perturbation.
    if perturbation_type == 0:
        # Random killing.
        inputs = pt.activation_knockout(inputs, perturbation_amount)
    elif perturbation_type == 1:
        # Activation noise.
        inputs = pt.activation_noise(inputs, perturbation_amount, int(inputs.get_shape()[0]))
    elif perturbation_type == 2:
        # Targeted killing.
        mask = tf.reshape(tf.tile(kill_mask[0]
                                  [int(np.prod(inputs.get_shape()[1:3])) * inputs.get_shape()[0]]),
                          [-1, int(inputs.get_shape()[1]), int(inputs.get_shape()[2]), int(inputs.get_shape()[3])])
        inputs = pt.activation_knockout_mask(inputs, perturbation_amount, mask)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)
    if use_batch_norm:
        inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    # Perturbation.
    if perturbation_type == 0:
        # Random killing.
        inputs = pt.activation_knockout(inputs, perturbation_amount)
    elif perturbation_type == 1:
        # Activation noise.
        inputs = pt.activation_noise(inputs, perturbation_amount, int(inputs.get_shape()[0]))
    elif perturbation_type == 2:
        # Targeted killing.
        mask = tf.reshape(tf.tile(kill_mask[1]
                                  [int(np.prod(inputs.get_shape()[1:3])) * inputs.get_shape()[0]]),
                          [-1, int(inputs.get_shape()[1]), int(inputs.get_shape()[2]), int(inputs.get_shape()[3])])
        inputs = pt.activation_knockout_mask(inputs, perturbation_amount, mask)

    return inputs


def block_layer(inputs, filters, use_batch_norm, bottleneck, block_fn, blocks,
                strides, training, name, data_format, perturbation_type,
                perturbation_amount, kill_mask):
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

        kill_mask: the kill masks for each convolution in the block layer
            (should have length 2 * blocks).

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
                    projection_shortcut, strides, data_format,
                    perturbation_type, perturbation_amount, kill_mask[0:2])

    for i in range(1, blocks):
        inputs = block_fn(inputs, filters, use_batch_norm, training, None, 1,
                      data_format, perturbation_type, perturbation_amount,
                      kill_mask[(2 * i):(2 * i + 2)])

    return tf.identity(inputs, name)


class Model(object):
    """Base class for building the Resnet Model."""
    
    def __init__(self, resnet_size, use_batch_norm, bottleneck, num_classes,
               num_filters, kernel_size, conv_stride, first_pool_size,
               first_pool_stride, block_sizes, block_strides, final_size,
               perturbation_type, perturbation_amount, kill_mask,
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

            kill_mask: here is a list with the same length as the number of layers,
                with the kill mask for each layer.

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
        self.perturbation_type = perturbation_type
        self.perturbation_amount = perturbation_amount
        self.kill_mask = kill_mask
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

            # Perturbation.
            if self.perturbation_type == 0:
                # Random killing.
                inputs = pt.activation_knockout(inputs, self.perturbation_amount)
            elif self.perturbation_type == 1:
                # Activation noise.
                inputs = pt.activation_noise(inputs, self.perturbation_amount, int(inputs.get_shape()[0]))
            elif self.perturbation_type == 2:
                # Targeted killing.
                mask = tf.reshape(tf.tile(self.kill_mask[0]
                                          [int(np.prod(inputs.get_shape()[1:3])) * inputs.get_shape()[0]]),
                                  [-1, int(inputs.get_shape()[1]), int(inputs.get_shape()[2]), int(inputs.get_shape()[3])])
                inputs = pt.activation_knockout_mask(inputs, self.perturbation_amount, mask)

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
                    name='block_layer{}'.format(i + 1), data_format=self.data_format,
                    perturbation_type=self.perturbation_type,
                    perturbation_amount=self.perturbation_amount,
                    kill_mask=self.kill_mask[(1 + 2 * (np.cumsum(self.block_sizes)[i] - self.block_sizes[i]))
                                            :(1 + 2 * np.cumsum(self.block_sizes)[i])])

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
