
from __future__ import print_function
import sys
import numpy as np

import tensorflow as tf

import experiments
from data import cifar_dataset
from models import nets

import os

import pickle
import time

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

num_perturbations = 5

def create_graph(perturbation_graph):
    ################################################################################################
    # Define training and validation datasets thorugh Dataset API
    ################################################################################################

    with open(opt.log_dir_base + opt.name + '/selectivity0.pkl', 'rb') as f:
        selectivity = pickle.load(f)

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

    select = [tf.placeholder(tf.float32, shape=(len(selectivity[1][k]))) for k in range(len(selectivity[1]))]

    robustness = tf.placeholder(tf.float32, shape=[num_perturbations, opt.dnn.layers]) #5 types of perturbation

    to_call = getattr(nets, opt.dnn.name + "_test")
    y, parameters = to_call(image, dropout_rate, select, opt, dataset.list_labels, robustness, perturbation_graph)

    # Accuracy
    y = tf.argmax(y, 1)
    im_prediction = tf.equal(y, y_)
    im_prediction = tf.cast(im_prediction, tf.float32)
    accuracy = tf.reduce_mean(im_prediction)

    num_iter_test = int((dataset.num_images_test) / opt.hyper.batch_size) - 1
    num_iter_train = int((dataset.num_images_training*opt.dataset.proportion_training_set)
                         / opt.hyper.batch_size) - 1

    return accuracy, y, im_prediction, handle, dropout_rate, robustness, select, \
           train_iterator_full, val_iterator_full, num_iter_test, num_iter_train
    ################################################################################################


def test_robustness(handle, dropout_rate, robustness, select,
                    opt, range_robustness, iterator_dataset, handle_dataset, num_iter, idx_rob):
    ####!@#@!#!@#!@#@!#!@#!@####!@#@!#!@#!@#@!#!@#!@
    ####!@#@!#!@#!@#@!#!@#!@
    ####!@#@!#!@#!@#@!#!@#!@
    #num_iter = 1

    #with open(opt.log_dir_base + opt.name + '/selectivity.pkl', 'rb') as f:
    #    selectivity = pickle.load(f)
    #    selectivity = selectivity[1]

    with open(opt.log_dir_base + opt.name + '/kernel0.pkl', 'rb') as f:
        selectivity = pickle.load(f)
        #selectivity = selectivity[::2]

    selectivity = [ np.argsort(selectivity[k])[np.random.randint(0, np.shape(selectivity[k])[0])]
                    for k in range(len(selectivity))]

    #print(selectivity)
    random_selectivity = []
    for k in range(len(selectivity)):
        idx = np.arange(len(selectivity[k]))
        np.random.shuffle(idx)
        random_selectivity.append(selectivity[k][idx])


    select_test = [np.zeros( [ len(selectivity[k]) ] ) for k in range(len(selectivity))]

    results_robustness = np.zeros([ opt.dnn.layers +
                                   1, len(range_robustness), 3])

    im_pred_gt = []
    im_y = []
    test_rob = np.zeros([num_perturbations,
                         opt.dnn.layers])
    sess.run(iterator_dataset.initializer)
    for _ in range(num_iter):
        dall = {}
        dall.update({i: d for i, d in zip(select, select_test)})
        dall.update({handle: handle_dataset, dropout_rate: opt.hyper.drop_test, robustness: test_rob})

        act_tmp, y_iter, im_pred_iter = \
            sess.run([accuracy, y, im_prediction], feed_dict=dall)
        im_pred_gt.append(im_pred_iter)
        im_y.append(y_iter)

    for layer in range(opt.dnn.layers+1):
        sys.stdout.flush()

        count = 0
        for noise_level in range_robustness:
            sess.run(iterator_dataset.initializer)

            test_rob = np.zeros([num_perturbations,
                                 opt.dnn.layers])
            select_test = [np.zeros([len(selectivity[k])]) for k in range(len(selectivity))]

            if layer < opt.dnn.layers:

                if idx_rob == 4:
                    min_idx = 0 #np.maximum(0,
                                #         int((len(selectivity[layer]) - int(
                                #             noise_level * len(selectivity[layer]))) / 2))

                    max_idx =  int(noise_level * len(selectivity[layer]))

                        #np.minimum(len(selectivity[layer]),
                        #                 int((len(selectivity[layer]) + int(
                        #                     noise_level * len(selectivity[layer]))) / 2))

                    select_test[layer][selectivity[layer][min_idx:max_idx]] = 1.0
                    test_rob[idx_rob, layer] = 1.0
                elif idx_rob == 2:
                    ####!@#@!#!@#!@#@!#!@#!@####!@#@!#!@#!@#@!#!@#!@
                    ####!@#@!#!@#!@#@!#!@#!@
                    ####!@#@!#!@#!@#@!#!@#!@
                    min_idx = 0
                    max_idx = int(noise_level * len(selectivity[layer]))

                    select_test[layer][random_selectivity[layer][min_idx:max_idx]] = 1.0
                    test_rob[4, layer] = 1.0

                else:
                    test_rob[idx_rob, layer] = noise_level
            else:
                if idx_rob == 4:
                    for l in range(len(selectivity)):
                        min_idx = 0
                        max_idx = int(noise_level * len(selectivity[l]))
                        select_test[l][selectivity[l][min_idx:max_idx]] = 1.0
                    test_rob[idx_rob, :] = 1.0
                elif idx_rob == 2:
                    ####!@#@!#!@#!@#@!#!@#!@
                    ####!@#@!#!@#!@#@!#!@#!@
                    ####!@#@!#!@#!@#@!#!@#!@
                    for l in range(len(selectivity)):
                        min_idx = 0
                        max_idx = int(noise_level * len(selectivity[l]))
                        select_test[l][random_selectivity[l][min_idx:max_idx]] = 1.0
                    test_rob[4, :] = 1.0
                else:
                    test_rob[idx_rob, :] = noise_level

            im_pred_min = []
            im_pred_max = []
            im_pred_ave = []
            for _ in range(num_iter):
                im_pred_min.append(np.zeros([opt.hyper.batch_size]))
                im_pred_ave.append(np.zeros([opt.hyper.batch_size]))
                im_pred_max.append(np.zeros([opt.hyper.batch_size]))

            #TRIALS = 1
            #for _ in range(TRIALS):
            sess.run(iterator_dataset.initializer)

            #im_pred = [np.zeros([opt.hyper.batch_size]) for _ in range(num_iter)]

            for idx in range(num_iter):

                dall = {}
                dall.update({i: d for i, d in zip(select, select_test)})
                dall.update({handle: handle_dataset,dropout_rate: opt.hyper.drop_test, robustness: test_rob})

                act_tmp, y_iter, im_pred_iter = \
                    sess.run([accuracy, y, im_prediction], feed_dict=dall)

                # Check which images have changed:
                im_pred_ave[idx][y_iter == im_y[idx]] = 1

            # MULTI TRIAL MIN-MAX ANALYSIS
            ''' 
            for idx in range(num_iter):
                im_pred_min[idx] = np.minimum(im_pred_min[idx], im_pred[idx])
                im_pred_ave[idx] += im_pred[idx]/TRIALS
                im_pred_max[idx] = np.maximum(im_pred_max[idx], im_pred[idx])
            '''


            acc_tmp_ave = 0.0
            for idx in range(num_iter):
                acc_tmp_ave += np.mean(im_pred_ave[idx])
            results_robustness[layer, count, 1] = acc_tmp_ave / num_iter

            total = 0
            acc_tmp_max = 0.0
            for idx in range(num_iter):
                acc_tmp_max += np.sum(im_pred_ave[idx][im_pred_gt[idx] == 1])
                total += np.sum(im_pred_gt[idx])
            if total == 0:
                results_robustness[layer, count, 0] = 0
            else:
                results_robustness[layer, count, 0] = acc_tmp_max / total

            total = 0
            acc_tmp_min = 0.0
            for idx in range(num_iter):
                acc_tmp_min += np.sum(im_pred_ave[idx][im_pred_gt[idx] == 0])
                total += np.sum((1-im_pred_gt[idx]))
            if total == 0:
                results_robustness[layer, count, 2] = 0
            else:
                results_robustness[layer, count, 2] = acc_tmp_min / total

            count += 1
    return results_robustness


''' 
ROBUSTNESS WITH ACCURACY!
def test_robustness(handle, dropout_rate, robustness,
                    opt, range_robustness, iterator_dataset, handle_dataset, num_iter, idx_rob):

    results_robustness = np.zeros([ opt.dnn.layers +
                                   1, len(range_robustness), 3])

    for layer in range(opt.dnn.layers+1):
        sys.stdout.flush()

        count = 0
        for noise_level in range_robustness:
            sess.run(iterator_dataset.initializer)

            test_rob = np.zeros([num_perturbations,
                                 opt.dnn.layers])

            if layer < opt.dnn.layers:
                test_rob[idx_rob, layer] = noise_level
            else:
                test_rob[idx_rob, :] = noise_level

            num_iter = 2
            im_pred_min = []
            im_pred_max = []
            im_pred_ave = []
            for _ in range(num_iter):
                im_pred_min.append(np.ones([opt.hyper.batch_size]))
                im_pred_ave.append(np.zeros([opt.hyper.batch_size]))
                im_pred_max.append(np.zeros([opt.hyper.batch_size]))

            TRIALS = 20
            for _ in range(TRIALS):
                sess.run(iterator_dataset.initializer)

                im_pred = []
                for _ in range(num_iter):
                    act_tmp, im_pred_iter = \
                        sess.run([accuracy, im_prediction], feed_dict={handle: handle_dataset,
                                                        dropout_rate: opt.hyper.drop_test,
                                                        robustness: test_rob})
                    im_pred.append(im_pred_iter)

                for idx in range(num_iter):
                    im_pred_min[idx] = np.minimum(im_pred_min[idx], im_pred[idx])
                    im_pred_ave[idx] += im_pred[idx]/TRIALS
                    im_pred_max[idx] = np.maximum(im_pred_max[idx], im_pred[idx])

            acc_tmp_min = 0.0
            acc_tmp_ave = 0.0
            acc_tmp_max = 0.0
            for idx in range(num_iter):
                acc_tmp_min += np.mean(im_pred_min[idx])
                acc_tmp_ave += np.mean(im_pred_ave[idx])
                acc_tmp_max += np.mean(im_pred_max[idx])

            results_robustness[layer, count, 0] = acc_tmp_min / num_iter
            results_robustness[layer, count, 1] = acc_tmp_ave / num_iter
            results_robustness[layer, count, 2] = acc_tmp_max / num_iter

            count += 1
    return results_robustness
'''

# For std
for cross in range(3):

    range_len = 7
    knockout_range = np.linspace(0.0, 1.0, num=range_len)
    noise_range = np.linspace(0.0, 1.0, num=range_len)
    # multiplicative_range = np.arange(0.0, 1.2, 0.2)

    noise_idx = [0, 3]
    knockout_idx = [1, 2, 4]
    results = [[[np.zeros([ opt.dnn.layers + 1, range_len, 3])]
                for j in range(2)] for i in range(num_perturbations)]

    flag_std = False
    for k in range(num_perturbations):
        #ONLY KNOCKOUT!
        if k != 4 and k != 2 and k != 3:
        #if k != 2:
            continue
        if k == 3:
            GG = 3
        else:
            GG = 4

        t0 = time.time()
        print("Perturbation: " + str(k))
        accuracy, y, im_prediction, handle, dropout_rate, robustness, select, \
        train_iterator_full, val_iterator_full, num_iter_test, num_iter_train \
            = create_graph(GG)
        ####!@#@!#!@#!@#@!#!@#!@
        ####!@#@!#!@#!@#@!#!@#!@
        ####!@#@!#!@#!@#@!#!@#!@

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
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

            if k in knockout_idx:
                range_perturbation = knockout_range
            else:
                range_perturbation = noise_range

            results[k][0][:] = test_robustness(handle, dropout_rate, robustness, select,
                                               opt, range_perturbation, train_iterator_full, train_handle_full,
                                               num_iter_train, k)

            results[k][1][:] = test_robustness(handle, dropout_rate, robustness, select,
                                               opt, range_perturbation, val_iterator_full, validation_handle_full,
                                               num_iter_test, k)

            #print(results)

        tf.reset_default_graph()
        t1 = time.time()
        print('Time: ', t1-t0)

    with open(opt.log_dir_base + opt.name + '/robustness' + str(cross) + '.pkl', 'wb') as f:
        pickle.dump(results, f)


print('Done :)')




