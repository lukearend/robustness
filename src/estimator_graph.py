"""Builds the graph for a forward pass."""

import resnet_model


RESNET_SIZE = 152


def forward_pass(x, is_training, params):
    resnet_size, block_sizes, bottleneck = {
        18: (18, [2, 2, 2, 2], False),
        34: (34, [3, 4, 6, 3], False),
        50: (50, [3, 4, 6, 3], True),
        101: (101, [3, 4, 23, 3], True),
        152: (152, [3, 8, 36, 3], True)}[RESNET_SIZE]
    num_filters = {
        'imagenet': 1000,
        'cifar10': 10}[params['dataset']]

    resnet = resnet_model.Model(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=params['num_classes'],
        num_filters=params['num_filters'],
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=block_sizes,
        block_strides=[2, 2, 2, 2],
        final_size=8 * params['num_filters'])

    y = resnet(x, is_training)

    return y
