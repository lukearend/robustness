
from __future__ import print_function
import sys
import numpy as np
import os
import tensorflow as tf

import experiments
from data import cifar_dataset
from models import nets

import pickle
import time

from numpy import linalg as LA

################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print(opt.name)
################################################################################################

MAX_SAMPLES = 50000

def create_graph():
    ################################################################################################
    # Define training and validation datasets thorugh Dataset API
    ################################################################################################

    # Initialize dataset and creates TF records if they do not exist
    dataset = cifar_dataset.Cifar10(opt)

    # No repeatable dataset for testing
    train_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='train', repeat=False)
    val_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='test', repeat=False)

    # Hadles to switch datasets
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, val_dataset_full.output_types, val_dataset_full.output_shapes)

    train_iterator_full = train_dataset_full.make_initializable_iterator()
    val_iterator_full = val_dataset_full.make_initializable_iterator()
    ################################################################################################


    ################################################################################################
    # DNN
    ################################################################################################

    # Get data from dataset
    image, y_ = iterator.get_next()
    image = tf.image.resize_images(image, [opt.hyper.image_size, opt.hyper.image_size])

    # Call DNN
    dropout_rate = tf.placeholder(tf.float32)
    to_call = getattr(nets, opt.dnn.name)
    y, parameters, activations = to_call(image, dropout_rate, opt, dataset.list_labels)

    # Accuracy
    im_prediction = tf.equal(tf.argmax(y, 1), y_)
    im_prediction = tf.cast(im_prediction, tf.float32)
    accuracy = tf.reduce_mean(im_prediction)

    num_iter_test = int((dataset.num_images_test) / opt.hyper.batch_size) - 1
    num_iter_train = int((dataset.num_images_training*opt.dataset.proportion_training_set)
                         / opt.hyper.batch_size) - 1

    return activations, accuracy, y_, handle, dropout_rate, train_iterator_full, val_iterator_full, num_iter_test, num_iter_train
    ################################################################################################


def get_activations(handle, accuracy, gt, dropout_rate, activations,
                    opt, iterator_dataset, handle_dataset, num_iter, seed):


    sess.run(iterator_dataset.initializer)

    np.random.seed(42 + seed)

    activations_out = []
    acc_tmp = 0.0
    for _ in range(num_iter):
        activations_tmp, act_tmp, labels_tmp_pre = \
            sess.run([activations, accuracy, gt], feed_dict={handle: handle_dataset,
                                            dropout_rate: opt.hyper.drop_test})
        acc_tmp += act_tmp

        labels_tmp = []
        for layer in range(len(activations_tmp)):
            labels_tmp.append(np.repeat(labels_tmp_pre, np.prod(np.shape(activations_tmp[layer])[1:-1])))

            activations_tmp[layer] = \
                np.reshape(activations_tmp[layer], (-1, np.shape(activations_tmp[layer])[-1]))


            num_samples = np.shape(activations_tmp[layer])[0]
            max_samples_per_iter = int(MAX_SAMPLES / num_iter)
            if num_samples > max_samples_per_iter:
                idx = np.random.permutation(num_samples)[:max_samples_per_iter]
                activations_tmp[layer] = activations_tmp[layer][idx, :]
                labels_tmp[layer] = labels_tmp[layer][idx]

        if activations_out == []:
            activations_out = activations_tmp
            labels_out = labels_tmp
        else:
            for layer in range(len(activations_tmp)):
                activations_out[layer] = np.append(activations_out[layer], activations_tmp[layer], axis=0)
                labels_out[layer] = np.append(labels_out[layer], labels_tmp[layer], axis=0)

    acc_out = acc_tmp / num_iter
    print("Accuracy: " + str(acc_out))
    return activations_out, acc_out, labels_out


t0 = time.time()
if not os.path.isfile(opt.log_dir_base + opt.name + '/activations0.pkl'):
    activations, accuracy, gt, handle, dropout_rate, train_iterator_full, val_iterator_full, num_iter_test, num_iter_train\
        = create_graph()

    results = [[[] for i in range(2)] for i in range(6)]

    with tf.Session() as sess:
        ################################################################################################
        # Set up checkpoints and data
        ################################################################################################

        saver = tf.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

        print("RESTORE")
        saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))

        ################################################################################################

        ################################################################################################
        # RUN TEST
        ################################################################################################
        train_handle_full = sess.run(train_iterator_full.string_handle())
        validation_handle_full = sess.run(val_iterator_full.string_handle())

        for cross in range(3):
            res, acc, gt_labels = get_activations(handle, accuracy, gt, dropout_rate, activations,
                                               opt, train_iterator_full, train_handle_full, num_iter_train, cross)
            with open(opt.log_dir_base + opt.name + '/activations' + str(cross) +'.pkl', 'wb') as f:
                pickle.dump(res, f, protocol=2)
            with open(opt.log_dir_base + opt.name + '/labels' + str(cross) +'.pkl', 'wb') as f:
                pickle.dump(gt_labels, f, protocol=2)
            with open(opt.log_dir_base + opt.name + '/accuracy' + str(cross) +'.pkl', 'wb') as f:
                pickle.dump(acc, f, protocol=2)

            res_test, acc, gt_labels_test = get_activations(handle, accuracy, gt, dropout_rate, activations,
                                           opt, val_iterator_full, validation_handle_full, num_iter_test, cross)

            with open(opt.log_dir_base + opt.name + '/activations_test' + str(cross) +'.pkl', 'wb') as f:
                pickle.dump(res_test, f, protocol=2)
            with open(opt.log_dir_base + opt.name + '/labels_test' + str(cross) + '.pkl', 'wb') as f:
                pickle.dump(gt_labels_test, f, protocol=2)
            with open(opt.log_dir_base + opt.name + '/accuracy_test' + str(cross) +'.pkl', 'wb') as f:
                pickle.dump(acc, f, protocol=2)

    tf.reset_default_graph()
    t1 = time.time()
    print('Time collect activations and kernel: ', t1-t0)

print('Done :)')
sys.exit()


