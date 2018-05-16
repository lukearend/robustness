import tensorflow as tf
from utils import summary as summ
import perturbations as pt
from numpy import *
from models.local_conv import reformat_weights
from keras import backend as K

import numpy as np

import sys

num_neurons = [96, 256,  384, 192]

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):

    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def Alexnet(x, opt, labels_id, dropout_rate):

    reuse = False

    global num_neurons

    parameters = []
    activations = []
    # conv1
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal(
            [5, 5, 3, int(num_neurons[0]*opt.dnn.neuron_multiplier[0])],
            stddev=5e-2/np.maximum(1, opt.dnn.neuron_multiplier[0]), dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=
                                 tf.constant(0.0, shape=[int(num_neurons[0]*opt.dnn.neuron_multiplier[0])]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        print(np.prod(conv1.shape[1:]))

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv1, opt)
        activations += [conv1]

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    with tf.name_scope('lrn1') as scope:
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        kernel = tf.get_variable(initializer=tf.truncated_normal(
            [5, 5, int(num_neurons[0]*opt.dnn.neuron_multiplier[0]),int(num_neurons[1]*opt.dnn.neuron_multiplier[1])],
            stddev=5e-2/np.maximum(1, opt.dnn.neuron_multiplier[1]), dtype=tf.float32), name='weights')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[int(num_neurons[1]*opt.dnn.neuron_multiplier[1])]), name='biases')

        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

        print(np.prod(conv2.shape[1:]))

        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv2, opt)
        activations += [conv2]

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.name_scope('lrn2') as scope:
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(pool2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # local3
    with tf.variable_scope('local3', reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.

        dim = int(prod(lrn2.get_shape()[1:]))
        pool_vec = tf.reshape(lrn2, [opt.hyper.batch_size, -1])

        nneurons = int(num_neurons[2]*opt.dnn.neuron_multiplier[2])
        weights = tf.get_variable(
            initializer=tf.truncated_normal([dim, nneurons],
            stddev=0.04/np.maximum(1, opt.dnn.neuron_multiplier[2]), dtype=tf.float32), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[nneurons]), name='biases')

        local3t = tf.nn.relu(tf.matmul(pool_vec, weights) + biases, name=scope.name)
        local3 = tf.nn.dropout(local3t, dropout_rate)

        print(np.prod(local3.shape[1:]))

        activations += [local3]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(local3, opt)

    # local4
    with tf.variable_scope('local4', reuse=reuse) as scope:
        weights = tf.get_variable(
            initializer=tf.truncated_normal([nneurons, int(num_neurons[3]*opt.dnn.neuron_multiplier[3])],
                stddev=0.04/np.maximum(1, opt.dnn.neuron_multiplier[3]), dtype=tf.float32), name='weights')

        biases = tf.get_variable(initializer=tf.constant(0.1,shape=[int(num_neurons[3]*opt.dnn.neuron_multiplier[3])]), name='biases')
        local4t = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        local4 = tf.nn.dropout(local4t, dropout_rate)

        print(np.prod(local4.shape[1:]))


        activations += [local4]
        parameters += [weights]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(local4, opt)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear', reuse=reuse) as scope:
        weights = tf.get_variable(
            initializer=tf.truncated_normal([int(num_neurons[3]*opt.dnn.neuron_multiplier[3]), len(labels_id)],
                                            stddev=1 / (float(num_neurons[3])), dtype=tf.float32),
                             name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.0, shape=[len(labels_id)]), name='biases')
        fc8 = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        activations += [fc8]
        summ.variable_summaries(weights, biases, opt)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters, activations


def Alexnet_test(x, opt, select, labels_id, dropout_rate, robustness, robustness_graph):

    parameters = []

    global num_neurons
    # conv1
    # conv(5, 5, 96, 4, 4, padding='VALID', name='conv1')
    # conv1
    with tf.variable_scope('conv1', reuse=False) as scope:

        kernel = tf.get_variable(initializer=tf.truncated_normal(
            [5, 5, 3, int(num_neurons[0] * opt.dnn.neuron_multiplier[0])],
            stddev=5e-2, dtype=tf.float32), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[int(num_neurons[0] * opt.dnn.neuron_multiplier[0])]),
                                 name='biases')
        '''
        reformatted_conv1W = reformat_weights(kernel, filter_width=5, filter_height=5,
                                              in_channels=3, out_channels=int(num_neurons[0] * opt.dnn.neuron_multiplier[0]),
                                              input_width=opt.hyper.image_size, input_height=opt.hyper.image_size,
                                              padding=5, stride=1)

        if robustness_graph == 0:
            reformatted_conv1W = pt.weight_noise(reformatted_conv1W, robustness[0][0])
        elif robustness_graph == 1:
            reformatted_conv1W = pt.weight_knockout(reformatted_conv1W, robustness[1][0])

        #print(x.shape)

        #

        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")
        #print(x.shape)
        conv1_in = K.local_conv2d(inputs=x, kernel=reformatted_conv1W,
                                  kernel_size=[5, 5], strides=[1, 1],
                                  output_shape=[opt.hyper.image_size, opt.hyper.image_size], data_format='channels_last')
                                  
        pre_activation = tf.nn.bias_add(conv1_in, biases)
        conv1 = tf.nn.relu(pre_activation)

        '''
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)


        #Activation perturbation
        if robustness_graph == 2:
            conv1 = pt.activation_knockout(conv1, robustness[2][0])
        elif robustness_graph == 3:
            conv1 = pt.activation_noise(conv1, robustness[3][0], opt.hyper.batch_size)
        elif robustness_graph == 4:
            ss = tf.reshape(tf.tile(select[0], [int(prod(conv1.get_shape()[1:3]))*opt.hyper.batch_size]),
                            [-1, int(conv1.get_shape()[1]), int(conv1.get_shape()[2]), int(conv1.get_shape()[3])])
            conv1 = pt.activation_knockout_mask(conv1, robustness[4][0], ss)

        parameters += [kernel, biases]
        summ.variable_summaries(kernel, biases, opt)
        summ.activation_summaries(conv1, opt)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    with tf.variable_scope('pool1') as scope:
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'SAME'
        maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        summ.activation_summaries(maxpool1, opt)


    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    with tf.variable_scope('lrn1') as scope:
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(maxpool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    with tf.variable_scope('conv2',reuse=False) as scope:

        kernel = tf.get_variable(initializer=tf.truncated_normal(
            [5, 5, int(num_neurons[0]*opt.dnn.neuron_multiplier[0]),int(num_neurons[1]*opt.dnn.neuron_multiplier[1])],
            stddev=5e-2, dtype=tf.float32), name='weights')
        biases = tf.get_variable(initializer=tf.constant(0.1, shape=[int(num_neurons[1]*opt.dnn.neuron_multiplier[1])]), name='biases')

        '''
        feature_size = int32(opt.hyper.image_size/2 - 1)
        reformatted_conv2W = reformat_weights(kernel, filter_width=5, filter_height=5,
                                              in_channels=int(num_neurons[0] * opt.dnn.neuron_multiplier[0]), 
                                              out_channels=int(num_neurons[1]* opt.dnn.neuron_multiplier[1]),
                                              input_width=feature_size, input_height=feature_size,
                                              padding=5, stride=1)

        #Synaptic perturbation
        if robustness_graph == 0:
            reformatted_conv2W = pt.weight_noise(reformatted_conv2W, robustness[0][1])
        if robustness_graph == 1:
            reformatted_conv2W = pt.weight_knockout(reformatted_conv2W, robustness[1][1])

        parameters += [kernel, biases]
        summ.variable_summaries(kernel, biases, opt)

        #conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)

        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        lrn1 = tf.pad(lrn1, paddings, "CONSTANT")
        conv2_in = K.local_conv2d(inputs=lrn1, kernel=reformatted_conv2W,
                                  kernel_size=[5, 5], strides=[1, 1],
                                  output_shape=[feature_size, feature_size], data_format='channels_last')

        pre_activation = tf.nn.bias_add(conv2_in, biases)
        conv2 = tf.nn.relu(pre_activation)

        '''
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)


        #Activation perturbation
        if robustness_graph == 2:
            conv2 = pt.activation_knockout(conv2, robustness[2][1])
        if robustness_graph == 3:
            conv2 = pt.activation_noise(conv2, robustness[3][1], opt.hyper.batch_size)
        if robustness_graph == 4:
            ss = tf.reshape(tf.tile(select[1], [int(prod(conv2.get_shape()[1:3]))*opt.hyper.batch_size]),
                            [-1, int(conv2.get_shape()[1]), int(conv2.get_shape()[2]), int(conv2.get_shape()[3])])
            conv2 = pt.activation_knockout_mask(conv2, robustness[4][1], ss)

        summ.activation_summaries(conv2, opt)


    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    with tf.variable_scope('pool2') as scope:
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'SAME'
        maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        summ.activation_summaries(maxpool2, opt)


    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    with tf.variable_scope('lrn2') as scope:
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(maxpool2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    aa = lrn2
    num_neurons_before_fc = int(prod(aa.get_shape()[1:]))
    print(num_neurons_before_fc)
    sys.stdout.flush()

    # fc6
    with tf.variable_scope('local3', reuse=False) as scope:

        flattened = tf.reshape(aa, [opt.hyper.batch_size, -1])
        nneurons = int(num_neurons[2] * opt.dnn.neuron_multiplier[2])
        W = tf.get_variable(
            initializer=tf.truncated_normal([num_neurons_before_fc, nneurons], stddev=0.04, dtype=tf.float32), name='weights')
        b = tf.get_variable(initializer=tf.constant(0.1, shape=[nneurons]), name='biases')

        if robustness_graph == 0:
            W = pt.weight_noise(W, robustness[0][2])
        if robustness_graph == 1:
            W = pt.weight_knockout(W, robustness[1][2])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flattened, W), b, name=scope.name))

        #Activation perturbation
        if robustness_graph == 2:
            fc6 = pt.activation_knockout(fc6, robustness[2][2])
        if robustness_graph == 3:
            fc6 = pt.activation_noise(fc6, robustness[3][2], opt.hyper.batch_size)
        if robustness_graph == 4:
            ss = tf.reshape(tf.tile(select[2], [opt.hyper.batch_size]),
                            [-1, int(fc6.get_shape()[1])])
            fc6 = pt.activation_knockout_mask(fc6, robustness[4][2], ss)

        summ.activation_summaries(fc6, opt)
        dropout6 = tf.nn.dropout(fc6, dropout_rate)

    # fc7
    with tf.variable_scope('local4', reuse=False) as scope:

        W = tf.get_variable(
            initializer=tf.truncated_normal([nneurons, int(num_neurons[3]*opt.dnn.neuron_multiplier[3])],
                                            stddev=0.04, dtype=tf.float32), name='weights')

        b = tf.get_variable(initializer=tf.constant(0.1,shape=[int(num_neurons[3]*opt.dnn.neuron_multiplier[3])]), name='biases')


        if robustness_graph == 0:
            W = pt.weight_noise(W, robustness[0][3])
        if robustness_graph == 1:
            W = pt.weight_knockout(W, robustness[1][3])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(dropout6, W), b, name=scope.name))

        #Activation perturbation
        if robustness_graph == 2:
            fc7 = pt.activation_knockout(fc7, robustness[2][3])
        if robustness_graph == 3:
            fc7 = pt.activation_noise(fc7, robustness[3][3], opt.hyper.batch_size)
        if robustness_graph == 4:
            ss = tf.reshape(tf.tile(select[3], [opt.hyper.batch_size]),
                            [-1, int(fc7.get_shape()[1])])
            fc7 = pt.activation_knockout_mask(fc7, robustness[4][3], ss)

        summ.activation_summaries(fc7, opt)
        dropout7 = tf.nn.dropout(fc7, dropout_rate)


    # fc8
    with tf.variable_scope('softmax_linear', reuse=False) as scope:
        W = tf.get_variable(initializer=tf.truncated_normal([int(num_neurons[3] * opt.dnn.neuron_multiplier[3]), len(labels_id)],
                                         dtype=tf.float32,
                                         stddev=1e-2), name='weights')
        b = tf.get_variable(initializer=tf.constant(0.0, shape=[len(labels_id)]), name='biases')

        if robustness_graph == 0:
            W = pt.weight_noise(W, robustness[0][4])
        if robustness_graph == 1:
            W = pt.weight_knockout(W, robustness[1][4])

        parameters += [W, b]
        summ.variable_summaries(W, b, opt)

        fc8 = tf.nn.bias_add(tf.matmul(dropout7, W), b, name=scope.name)
        summ.activation_summaries(fc8, opt)

    return fc8, parameters

