import numbers

import numpy as np
import tensorflow as tf
# from tensorflow.python.eager import context
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.ops import array_ops, random_ops, math_ops


def unscaled_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):  # from tensorflow/pyton/ops/nn_ops/dropout
    """Computes dropout where the output is not scaled by the dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Args:
      x: A floating point tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      name: A name for this operation (optional).

    Returns:
      A Tensor of the same shape of `x`.

    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
        point tensor.
    """
    with ops.name_scope(name, "unscaled_dropout", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob,
                                          dtype=x.dtype,
                                          name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                   seed=seed,
                                                   dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        # if context.in_graph_mode():
        #    ret.set_shape(x.get_shape())
        return ret


def unscaled_dropout_mask(x, keep_prob, mask, noise_shape=None, seed=None, name=None):  # from tensorflow/pyton/ops/nn_ops/dropout
    """Computes dropout where the output is not scaled by the dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Args:
      x: A floating point tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      name: A name for this operation (optional).

    Returns:
      A Tensor of the same shape of `x`.

    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
        point tensor.
    """
    with ops.name_scope(name, "unscaled_dropout", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        # if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
        #     raise ValueError("keep_prob must be a scalar tensor or a float in the "
        #                      "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob,
                                          dtype=x.dtype,
                                          name="keep_prob")
        mask = ops.convert_to_tensor(mask,
                                          dtype=x.dtype,
                                          name="mask")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape,
                                                   seed=seed,
                                                   dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        binary_tensor = binary_tensor * mask + (1.0 - mask)
        ret = x * binary_tensor
        # if context.in_graph_mode():
        #    ret.set_shape(x.get_shape())
        return ret


def activation_knockout(x, knockout_prob, noise_shape=None, seed=None, name=None):
    return unscaled_dropout(x, 1 - knockout_prob, noise_shape=noise_shape, seed=seed, name=name)


def activation_knockout_mask(x, knockout_prob, mask, noise_shape=None, seed=None, name=None):
    return unscaled_dropout_mask(x, 1 - knockout_prob, mask, noise_shape=noise_shape, seed=seed, name=name)


def activation_knockout_compensated(x, knockout_prob):
    return tf.nn.dropout(x, 1 - knockout_prob)


def activation_noise(x, variance_proportion, batch_size, variance_excluded_axes=(0,), seed=None, name=None):
    with ops.name_scope(name, "activation-gaussian_noise", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        variance_proportion = ops.convert_to_tensor(variance_proportion,
                                                    dtype=x.dtype,
                                                    name="variance_proportion")
        # Do nothing if we know variance_proportion == 0
        if tensor_util.constant_value(variance_proportion) == 0:
            return x

        x_shape = x.get_shape().as_list()
        x_shape[0] = batch_size
        # we take the variance in the activation vector to a single input
        variance_axes = set(range(len(x_shape))) - set(variance_excluded_axes)
        variance_axes = list(variance_axes)
        _, gaussian_variance = tf.nn.moments(x, axes=variance_axes)
        gaussian_stddev = tf.sqrt(gaussian_variance)
        gaussian_stddev = variance_proportion * gaussian_stddev

        if len(variance_excluded_axes) == 0:  # if no axes are excluded, we have to make stddev into 1D
            gaussian_stddev = tf.expand_dims(gaussian_stddev, axis=0)

        stddev_repetitions = np.prod([x_shape[axis] for axis in variance_axes])
        noise_tensor = tf.random_normal(
            # sample all the `#inputs * #features` numbers
            [np.prod(x_shape)],
            mean=0,
            # stack standard deviations so that every `batch size` times, the same stddev is repeated
            stddev=tf.tile(gaussian_stddev, [stddev_repetitions]),
            seed=seed, dtype=x.dtype)
        # have to hack the following a little because tf.reshape does not give us any way to influence the ordering.
        # before this, the noise_tensor has related features separated by batch size.
        # E.g. for a batch size of 4, the noise_tensor would be something like [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, ...].
        # What we want however is [[1, 1, 1, ...], [2, 2, 2, ...], [3, 3, 3, ...], ...].
        # So we reshape the noise_tensor into [[1, 2, 3, 4], [1, 2, 3, 4], ...] and take the transpose of that.
        noise_tensor = tf.transpose(tf.reshape(noise_tensor, list(reversed(x_shape))))
        ret = x + noise_tensor
        return ret


def activation_multiplicative_shift(x, factor, seed=None, name=None):
    with ops.name_scope(name, "activation-multiplicative_shift", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        factor = ops.convert_to_tensor(factor,
                                       dtype=x.dtype,
                                       name="factor")
        # Do nothing if we know factor == 0
        if tensor_util.constant_value(factor) == 0:
            return x

        random_tensor = tf.random_uniform(array_ops.shape(x),
                                          minval=1 - factor / 2,
                                          maxval=1 + factor / 2,
                                          seed=seed, dtype=x.dtype)
        ret = math_ops.multiply(x, random_tensor)
        return ret


def weight_dropout(weight_variable, keep_prob, seed=None, name="weight-dropout"):
    return unscaled_dropout(weight_variable, keep_prob=keep_prob, seed=seed, name=name)


def weight_knockout(weight_variable, knockout_prob, seed=None, name=None):
    return weight_dropout(weight_variable, 1 - knockout_prob, seed=seed, name=name)


def weight_noise(weight_variable, variance_proportion, seed=None, name="weight-gaussian_noise"):
    return activation_noise(weight_variable, variance_proportion, batch_size=weight_variable.get_shape().as_list()[0],
                            variance_excluded_axes=(), seed=seed, name=name)


def weight_multiplicative_shift(weight_variable, factor, seed=None, name="weight-synapse-multiplicative_shift"):
    return activation_multiplicative_shift(weight_variable, factor=factor, seed=seed, name=name)
