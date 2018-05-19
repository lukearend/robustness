"""Builds the graph for a forward pass."""

import resnet_model
import resnet_model_test

import resnet_model_orig


def forward_pass(x, is_training, params):
    # if params['dataset'] == 'imagenet':
    #     resnet = resnet_model.Model(
    #         resnet_size=18,
    #         use_batch_norm=params['use_batch_norm'],
    #         bottleneck=False,
    #         num_classes=1000,
    #         num_filters=params['num_filters'],
    #         kernel_size=7,
    #         conv_stride=2,
    #         first_pool_size=3,
    #         first_pool_stride=2,
    #         block_sizes=[2, 2, 2, 2],
    #         block_strides=[2, 2, 2, 2],
    #         final_size=params['num_filters'] * 8)
    # elif params['dataset'] == 'cifar10':
    #     resnet = resnet_model.Model(
    #         resnet_size=20,
    #         use_batch_norm=params['use_batch_norm'],
    #         bottleneck=False,
    #         num_classes=10,
    #         num_filters=params['num_filters'],
    #         kernel_size=3,
    #         conv_stride=1,
    #         first_pool_size=0,
    #         first_pool_stride=1,
    #         block_sizes=[3, 3, 3],
    #         block_strides=[1, 2, 2],
    #         final_size=params['num_filters'] * 4,
    #         data_format='channels_last' if not params['use_batch_norm'] else None)

    # y, activations = resnet(x, is_training)

    resnet = resnet_model_orig.Model(
        resnet_size=50,
        bottleneck=True,
        num_classes=1000,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=[3, 4, 6, 3],
        block_strides=[2, 2, 2, 2],
        final_size=2048)

    y = resnet(x, is_training)

    return y

def forward_pass_test(x, params, perturbation_type, perturbation_amount, kill_mask):
    # For perturbation experiment.
    if params['dataset'] == 'imagenet':
        resnet = resnet_model_test.Model(
            resnet_size=18,
            use_batch_norm=params['use_batch_norm'],
            bottleneck=False,
            num_classes=1000,
            num_filters=params['num_filters'],
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=[2, 2, 2, 2],
            block_strides=[2, 2, 2, 2],
            final_size=params['num_filters'] * 8,
            perturbation_type=perturbation_type,
            perturbation_amount=perturbation_amount,
            kill_mask=kill_mask)
    elif params['dataset'] == 'cifar10':
        resnet = resnet_model_test.Model(
            resnet_size=20,
            use_batch_norm=params['use_batch_norm'],
            bottleneck=False,
            num_classes=10,
            num_filters=params['num_filters'],
            kernel_size=3,
            conv_stride=1,
            first_pool_size=0,
            first_pool_stride=1,
            block_sizes=[3, 3, 3],
            block_strides=[1, 2, 2],
            final_size=params['num_filters'] * 4,
            perturbation_type=perturbation_type,
            perturbation_amount=perturbation_amount,
            kill_mask=kill_mask,
            data_format='channels_last' if not params['use_batch_norm'] else None)

    y = resnet(x, False)

    return y
