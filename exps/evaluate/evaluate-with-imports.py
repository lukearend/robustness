"""Evaluate a model."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--scale_factor', type=float, required=True)
parser.add_argument('--use_batch_norm', type=bool, required=True)
parser.add_argument('--dataset', type-str, required=True)
FLAGS = parser.parse_args()

sys.path.append('/om/user/larend/robust/src')
import estimator


def main():
    num_filters = {
        0.25: 16,
        0.5: 32,
        1: 64,
        2: 128,
        4: 256}[FLAGS.scale_factor]

    data_dir = {
        'cifar10': '/om/user/larend/data/cifar-10-tfrecords',
        'imagenet': '/om/user/larend/data/imagenet-tfrecords'}[FLAGS.dataset]


    model = Estimator(
        model_dir=FLAGS.model_dir,
        params={
            'batch_size': 100,
            'dataset': FLAGS.dataset,
            'use_batch_norm': FLAGS.use_batch_norm,
            'num_filters': {
                0.25: 16,
                0.5: 32,
                1: 64,
                2: 128,
                4: 256}[FLAGS.scale_factor]},
        tf_random_seed=12345)

    model.evaluate(
        data_dir={
            'cifar10': '/om/user/larend/data/cifar-10-tfrecords',
            'imagenet': '/om/user/larend/data/imagenet-tfrecords'}[FLAGS.dataset])


if __name__ == '__main__':
    main()
