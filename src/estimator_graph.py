"""Builds the graph for a forward pass."""

import tensorflow as tf

# import resnet_model


# RESNET_SIZE = 18


# def forward_pass(x, is_training, params):
#     resnet_size, block_sizes, bottleneck = {
#         18: (18, [2, 2, 2, 2], False),
#         34: (34, [3, 4, 6, 3], False),
#         50: (50, [3, 4, 6, 3], True),
#         101: (101, [3, 4, 23, 3], True),
#         152: (152, [3, 8, 36, 3], True)}[RESNET_SIZE]
#     num_classes = {
#         'imagenet': 1000,
#         'cifar10': 10}[params['dataset']]
#     num_filters = params['num_filters']
#     final_size = {
#         True: num_filters * 2 ** (len(block_sizes) - 1) * 4,
#         False: num_filters * 2 ** (len(block_sizes) - 1)}[bottleneck]

#     resnet = resnet_model.Model(
#         resnet_size=resnet_size,
#         bottleneck=bottleneck,
#         num_classes=num_classes,
#         num_filters=num_filters,
#         kernel_size=7,
#         conv_stride=2,
#         first_pool_size=3,
#         first_pool_stride=2,
#         block_sizes=block_sizes,
#         block_strides=[2, 2, 2, 2],
#         final_size=final_size)

#     y = resnet(x, is_training)

#     return y


def forward_pass(x, is_training, params):
    x = tf.layers.conv2d(x, filters=20, kernel_size=5, strides=1)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    x = tf.layers.conv2d(x, filters=50, kernel_size=5, strides=1)
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    size = x.get_shape().as_list()
    x = tf.reshape(x, [-1, size[1] * size[2] * size[3]])
    x = tf.layers.dense(x, 500)
    x = tf.nn.relu(x)
    x = tf.layers.dense(x, 10)

    return x
