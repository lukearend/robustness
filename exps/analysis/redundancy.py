
from __future__ import print_function
import sys
import numpy as np
import os

import pickle
import time

import argparse

from numpy import linalg as LA

################################################################################################
# Read experiment to run
################################################################################################

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--pickle_dir', type=str, required=True)
FLAGS = parser.parse_args()

###############################################################################

MAX_SAMPLES = 50000

def pca(data, energy):
    data = data - data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    th = np.zeros(len(energy))
    for en, idx_en in zip(energy,range(len(energy))):
        total = np.sum(eigenvalues)
        accum = 0
        k =  1
        while accum < en:
            accum += eigenvalues[-k]/total
            k += 1
        th[idx_en] = k - 1
    return th, eigenvalues, eigenvectors


def non_zero(eigenvalues):
    total = np.sum(eigenvalues)
    accum = 0
    k = 0
    while accum < 0.05:
        accum += eigenvalues[k] / total
        k += 1
    th = k

    return th


def selectivity(res, gt, res_test, gt_test):

    num_neurons = np.shape(np.mean(res, axis=0))[0]

    ave_c = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_c_test = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_all = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    ave_all_test = np.zeros([num_neurons, np.shape(np.unique(gt, axis=0))[0]])
    for k in np.unique(gt, axis=0).tolist():
        ave_c[:, k] = np.mean(res[gt == k], axis=0)
        ave_c_test[:, k] = np.mean(res_test[gt_test == k], axis=0)

        ave_all[:, k] = np.mean(res[gt != k], axis=0)
        ave_all_test[:, k] = np.mean(res_test[gt_test != k], axis=0)

    idx_max = np.argmax(ave_c, axis=1)
    sel = np.zeros([num_neurons])
    sel_test = np.zeros([num_neurons])

    for idx_k, k in enumerate(idx_max.tolist()):
        sel[idx_k] = (ave_c[idx_k, k] - ave_all[idx_k, k])/(ave_c[idx_k, k] + ave_all[idx_k, k])
        sel_test[idx_k] = (ave_c_test[idx_k, k] - ave_all_test[idx_k, k]) / (ave_c_test[idx_k, k] + ave_all_test[idx_k, k])

    sel_gen = ((sel-sel_test)**2)

    return sel, sel_test, sel_gen


def Kernel(res, res_test):

    num_neurons = np.shape(np.mean(res, axis=0))[0]

    for k in range(num_neurons):
        norm = LA.norm(res[:, k], axis=0)
        print(norm)
        norm_test = LA.norm(res_test[:, k], axis=0)
        print(norm_test)

        norm[np.where(norm == 0.)] = 1.
        norm_test[np.where(norm_test == 0.)] = 1.

        res[:, k] = res[:, k] / norm
        res_test[:, k] = res_test[:, k] / norm_test

    K = np.zeros([num_neurons, num_neurons])
    K_test = np.zeros([num_neurons, num_neurons])
    for i in range(num_neurons):
        for j in range(i, num_neurons):
            K[i, j] = np.sum((res[:, i] - res[:, j])**2)
            K[j, i] = K[i, j]
            K_test[i, j] = np.sum((res_test[:, i] - res_test[:, j])**2)
            K_test[j, i] = K[i, j]

    return K, K_test


def get_NN(K):

    num_neurons = np.shape(K)[0]
    res = np.zeros([num_neurons])
    for i in range(num_neurons):
       res[i] = sum(K[i, j] < 0.3 for j in range(num_neurons)) - 1

    return np.average(res), np.std(res)


t0 = time.time()
if not os.path.isfile(FLAGS.pickle_dir + '/activations0.pkl'):

    print("ERROR")
    sys.exit()

for cross in range(3):
    t0 = time.time()
    results = [[[] for i in range(2)] for i in range(6)]

    with open(FLAGS.pickle_dir + '/activations' + str(cross) +'.pkl', 'rb') as f:
        res = pickle.load(f)
    with open(FLAGS.pickle_dir + '/labels' + str(cross) +'.pkl', 'rb') as f:
        gt_labels = pickle.load(f)
    with open(FLAGS.pickle_dir + '/accuracy' + str(cross) +'.pkl', 'rb') as f:
        acc = pickle.load(f)


    with open(FLAGS.pickle_dir + '/activations_test' + str(cross) +'.pkl', 'rb') as f:
        res_test = pickle.load(f)
    with open(FLAGS.pickle_dir + '/labels_test' + str(cross) +'.pkl', 'rb') as f:
        gt_labels_test = pickle.load(f)
    with open(FLAGS.pickle_dir + '/accuracy_test' + str(cross) +'.pkl', 'rb') as f:
        acc_test = pickle.load(f)


    kernel = []
    kernel_test = []
    res_NN = [[[] for i in range(2)] for j in range(2)]
    for layer in range(len(res)):
        print('layer: {}\tres: {}\tres_test: {}'.format(layer, res[layer].shape, res_test[layer].shape))
        k_tmp, k_test_tmp = Kernel(res[layer], res_test[layer])
        kernel.append(k_tmp)
        kernel_test.append(k_test_tmp)

        mean_NN, std_NN = get_NN(k_tmp)
        res_NN[0][0].append(mean_NN)
        res_NN[0][1].append(std_NN)

        mean_NN, std_NN = get_NN(k_test_tmp)
        res_NN[1][0].append(mean_NN)
        res_NN[1][1].append(std_NN)


    with open(FLAGS.pickle_dir + '/kernel' + str(cross) +'.pkl', 'wb') as f:
        pickle.dump(kernel, f, protocol=2)

    with open(FLAGS.pickle_dir + '/kernel_test' + str(cross) +'.pkl', 'wb') as f:
        pickle.dump(kernel_test, f, protocol=2)

    with open(FLAGS.pickle_dir + '/NN' + str(cross) +'.pkl', 'wb') as f:
        pickle.dump(res_NN, f, protocol=2)

    for layer in range(len(res)):
        k, spectrum, W = pca(res[layer], (0.95, 0.8))
        results[0][0].append(k)
        results[1][0].append(spectrum)
        results[2][0].append(W)
        results[3][0].append(acc)
        k = non_zero(spectrum)
        results[5][0].append(k)


    for layer in range(len(res_test)):
        k, spectrum, W = pca(res_test[layer], (0.95, 0.8))
        results[0][1].append(k)
        results[1][1].append(spectrum)
        results[2][1].append(W)
        results[3][1].append(acc_test)
        k = non_zero(spectrum)
        results[5][1].append(k)

    for layer in range(len(res_test)):
        results[4][1].append(np.shape(res_test[layer])[-1])

    results_selectivity = [[] for i in range(3)]
    for layer in range(len(res)):
        sel, sel_test, sel_gen = selectivity(res[layer], gt_labels[layer], res_test[layer], gt_labels_test[layer])
        results_selectivity[0].append(sel)
        results_selectivity[1].append(sel_test)
        results_selectivity[2].append(sel_gen)

    t1 = time.time()
    print('Time: ', t1-t0)


    with open(FLAGS.pickle_dir + '/redundancy' + str(cross) +'.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)

    with open(FLAGS.pickle_dir + '/selectivity' + str(cross) +'.pkl', 'wb') as f:
        pickle.dump(results_selectivity, f, protocol=2)


print('Done :)')
