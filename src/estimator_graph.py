"""Builds the graph for a forward pass."""

import resnet_model


def forward_pass(x, is_training, params):
    resnet = resnet_model.Model(
        resnet_size=18,
        bottleneck=False,
        num_classes=1000,
        num_filters=params['num_filters'],
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=[2, 2, 2, 2],
        block_strides=[2, 2, 2, 2],
        final_size=512)

    y = resnet(x, is_training)

    return y
