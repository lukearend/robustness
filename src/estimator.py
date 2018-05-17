"""A wrapper class for the model."""

import os
import time
import math
import pickle

import numpy as np

import tensorflow as tf

import estimator_utils
import estimator_datasets
import estimator_fns


# Get log messages from tf.Estimator.
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Estimator(object):
    """A class that wraps up the estimator."""
    DEFAULT_PARAMS = {
        'initial_learning_rate': 0.1,
        'learning_rate_decay_factor': 0.1,
        'num_epochs_per_decay': 30,
        'max_epochs': 120,
        'train_with_distortion': True,
        'momentum': 0.9,
        'batch_size': 256,
        'weight_decay': 0.0001,
        'dataset': 'imagenet',
        'use_batch_norm': True,
        'num_filters': 64
    }

    def __init__(self, model_dir='/tmp', params={}, tf_random_seed=12435):
        self.model_dir = model_dir
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(params)
        self.tf_random_seed = tf_random_seed

        self.session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(
                # Use only as much GPU memory as needed based on
                # runtime allocations.
                allow_growth=True,
                # Force all CPU tensors to be allocated with Cuda
                # pinned memory. May hurt host performance if model is
                # extremely large, as all Cuda pinned memory is
                # unpageable.
                force_gpu_compatible=True))

    def train(self,
              data_dir='/tmp',
              num_gpus=1,
              save_summary_steps=100,
              save_checkpoint_and_validate_secs=None,
              save_checkpoint_and_validate_steps=None,
              keep_checkpoint_max=5,
              keep_checkpoint_every_n_hours=10000,
              early_stopping_epochs=None,
              log_every_n_steps=10):
        """Train the model.

        Args:
            data_dir: directory in which to look for training/validation
                      data.
            num_gpus: number of GPUs to use.
            save_summary_steps: save a summary every this many steps.
            save_checkpoint_and_validate_secs: save a checkpoint and run
                                               validation every this
                                               many seconds.
            save_checkpoint_and_validate_steps: save a checkpoint and run
                                                validation every this
                                                many steps.
            keep_checkpoint_max: keep this many checkpoints max.
            keep_checkpoint_every_n_hours: keep a permanent checkpoint
                                           every this many hours.
            log_every_n_steps: log steps/sec and chosen Tensors every
                               this many steps.
        """
        # Set defaults for saving checkpoints and validating.
        if (save_checkpoint_and_validate_secs is not None and
            save_checkpoint_and_validate_steps is not None):
            raise ValueError("Please set only one of '"
                             "'save_checkpoint_and_validate_secs' or "
                             "'save_checkpoint_and_validate_steps'.")
        elif (save_checkpoint_and_validate_secs is None and
              save_checkpoint_and_validate_steps is None):
            save_checkpoint_and_validate_secs = 600

        # Configure and build the model.
        model_fn = estimator_fns.get_model_fn(
            num_gpus,
            keep_checkpoint_max=keep_checkpoint_max,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

        config = tf.estimator.RunConfig().replace(
            save_summary_steps=save_summary_steps,
            save_checkpoints_secs=save_checkpoint_and_validate_secs,
            save_checkpoints_steps=save_checkpoint_and_validate_steps,
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=config,
                                       params=self.params)

        # Configure training.
        train_input_fn = lambda: estimator_fns.input_fn(
            tf.estimator.ModeKeys.TRAIN,
            data_dir,
            self.params,
            num_gpus)

        validation_input_fn = lambda: estimator_fns.input_fn(
            tf.estimator.ModeKeys.EVAL,
            data_dir,
            self.params,
            num_gpus)

        if early_stopping_epochs is not None:
            early_stopping_rounds = int(
                estimator_utils.get_num_examples_per_epoch(
                    self.params['dataset'], tf.estimator.ModeKeys.TRAIN) /
                self.params['batch_size'] *
                early_stopping_epochs)
        else:
            early_stopping_rounds = None

        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=validation_input_fn,
            every_n_steps=1,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric='precision/precision_at_one',
            early_stopping_metric_minimize=False)

        examples_per_second_hook = estimator_utils.ExamplesPerSecondHook(
            self.params['batch_size'],
            every_n_steps=log_every_n_steps)

        logging_hook = tf.train.LoggingTensorHook(
            {'loss': 'loss',
             'cross_entropy_loss': 'cross_entropy_loss',
             'weight_loss': 'weight_loss',
             'precision_at_one': 'precision_at_one',
             'precision_at_five': 'precision_at_five'},
            every_n_iter=log_every_n_steps)

        hooks = estimator_utils.replace_monitors_with_hooks(
            [validation_monitor, examples_per_second_hook, logging_hook],
            model)

        max_steps = int(
            estimator_utils.get_num_examples_per_epoch(
                self.params['dataset'], tf.estimator.ModeKeys.TRAIN) /
            self.params['batch_size'] *
            self.params['max_epochs'])

        # Train the model.
        model.train(input_fn=train_input_fn,
                    hooks=hooks,
                    max_steps=max_steps)

    def evaluate(self, data_dir='/tmp', num_gpus=1):
        """Evaluate the model.

        Args:
            data_dir: directory in which to look for validation data.
            num_gpus: number of GPUs to use.
        """
        # Configure and build the model.
        model_fn = estimator_fns.get_model_fn(num_gpus)

        config = tf.estimator.RunConfig().replace(
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=config,
                                       params=self.params)

        input_fn = lambda: estimator_fns.input_fn(tf.estimator.ModeKeys.EVAL,
                                               data_dir,
                                               self.params,
                                               num_gpus)

        model.evaluate(input_fn=input_fn)

    def activations(self, num_layers, data_dir='/tmp', split='train', num_gpus=1):
        """Extract activations to a dataset.

        Args:
            data_dir: directory in which to look for validation data.
            split: one of 'train' or 'validation'.
            num_gpus: number of GPUs to use.
        """
        # Set seed!
        imagenet_train_predict_shuffle_seed = int(time.time())

        # First, loop through the dataset and read out labels.
        _, label_batch = estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
                                                data_dir,
                                                self.params,
                                                reading_labels=True,
                                                predict_split=split,
                                                imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                                imagenet_train_predict_partial=True)
        labels = None
        with tf.Session() as sess:
            while True:
                try:
                    labels_tmp = sess.run(label_batch)
                    if labels is None:
                        labels = labels_tmp
                    else:
                        labels = np.append(labels, labels_tmp, axis=0)
                except tf.errors.OutOfRangeError:
                    break

        # Extract activations.
        model_fn = estimator_fns.get_model_fn(num_gpus)

        config = tf.estimator.RunConfig().replace(
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=config,
                                       params=self.params)
        input_fn = lambda: estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
                                               data_dir,
                                               self.params,
                                               num_gpus=num_gpus,
                                               predict_split=split,
                                               imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                               imagenet_train_predict_partial=True)
        predictions = model.predict(input_fn)

        # Loop through predictions and store them in a numpy array;
        # done in batches for memory efficiency.
        if self.params['dataset'] == 'cifar10':
            extraction_batch_size = 100
        elif self.params['dataset'] == 'imagenet':
            extraction_batch_size = 25
        num_predictions = len(labels)
        num_iterations = math.ceil(num_predictions / extraction_batch_size)
        predicted_labels = np.zeros(np.shape(labels))
        for i in range(num_iterations):
            tic = time.time()

            if i == 0:
                activations_out = list(range(num_layers))
                labels_out = list(range(num_layers))

            for j in range(extraction_batch_size):
                p = next(predictions)
                predicted_labels[i * extraction_batch_size + j] = p['classes']

                if j == 0:
                    activations_batch = list(range(num_layers))
                    labels_batch = list(range(num_layers))

                for layer in range(num_layers):
                    layer_activations = np.reshape(p[layer], (-1, np.shape(p[layer])[-1]))
                    layer_labels = np.repeat(labels[i * extraction_batch_size + j],
                                             np.prod(np.shape(p[layer])[:-1]))

                    if j == 0:
                        activations_batch[layer] = layer_activations
                        labels_batch[layer] = layer_labels
                    else:
                        activations_batch[layer] = np.append(activations_batch[layer], layer_activations, axis=0)
                        labels_batch[layer] = np.append(labels_batch[layer], layer_labels, axis=0)

            MAX_SAMPLES = 50000
            max_samples_per_iteration = int(MAX_SAMPLES / num_iterations)
            for layer in range(num_layers):
                num_samples = np.shape(labels_batch[layer])[0]
                if num_samples > max_samples_per_iteration:
                    idx = np.random.permutation(num_samples)[:max_samples_per_iteration]
                    activations_batch[layer] = activations_batch[layer][idx, :]
                    labels_batch[layer] = labels_batch[layer][idx]

                if i == 0:
                    activations_out[layer] = activations_batch[layer]
                    labels_out[layer] = labels_batch[layer]
                else:
                    activations_out[layer] = np.append(activations_out[layer], activations_batch[layer], axis=0)
                    labels_out[layer] = np.append(labels_out[layer], labels_batch[layer], axis=0)

            toc = time.time()
            print('extraction iteration {}: {} sec'.format(i, toc - tic))

        accuracy = np.mean(np.equal(labels, predicted_labels))

        return activations_out, labels_out, accuracy

    def robustness(self,
        perturbation_type,
        perturbation_amount,
        kernel_filename,
        data_dir='/tmp',
        split='train',
        num_gpus=1):
        """Test robustness of a model.

        Args:
            perturbation_type, perturbation_amount, kernel_filename:
                settings for testing robustness.
            data_dir: directory in which to look for validation data.
            split: one of 'train' or 'validation'.
            num_gpus: number of GPUs to use.
        """
        # Set seed!
        imagenet_train_predict_shuffle_seed = int(time.time())

        # Load kernel file and use it to compute kill_mask.
        with open(kernel_filename, 'rb') as f:
            kernel = pickle.load(f)
        num_layers = len(kernel)

        if perturbation_type in [0, 2]:
            kill_mask = []
            for layer in range(num_layers):
                num_neurons = np.shape(kernel[layer])[0]
                num_to_kill = int(perturbation_amount * num_neurons)

                if perturbation_type == 0:
                    # Select neurons for killing at random.
                    indices = np.arange(num_neurons)
                    np.random.shuffle(indices)
                    neurons_to_kill = indices[:num_to_kill]
                elif perturbation_type == 2:
                    # Pick a random neuron and find its nearest neighbors to kill.
                    start_neuron = np.random.randint(0, num_neurons)
                    nearest_neighbors = np.argsort(kernel[layer][start_neuron])
                    neurons_to_kill = nearest_neighbors[:num_to_kill]

                kill_mask.append(np.zeros(num_neurons))
                kill_mask[layer][neurons_to_kill] = 1.
        else:
            kill_mask = [None for _ in range(len(kernel))]

        print(kill_mask)

        # First, loop through the dataset and read out labels.
        _, label_batch = estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
                                                data_dir,
                                                self.params,
                                                reading_labels=True,
                                                predict_split=split,
                                                imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                                imagenet_train_predict_partial=True)
        labels = None
        with tf.Session() as sess:
            while True:
                try:
                    labels_tmp = sess.run(label_batch)
                    if labels is None:
                        labels = labels_tmp
                    else:
                        labels = np.append(labels, labels_tmp, axis=0)
                except tf.errors.OutOfRangeError:
                    break

        # Test robustness.
        model_fn = estimator_fns.get_model_fn(
            num_gpus,
            test_robustness=True,
            perturbation_type=perturbation_type,
            perturbation_amount=perturbation_amount,
            kill_mask=kill_mask)

        config = tf.estimator.RunConfig().replace(
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=config,
                                       params=self.params)
        input_fn = lambda: estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
                                               data_dir,
                                               self.params,
                                               num_gpus=num_gpus,
                                               predict_split=split,
                                               imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
                                               imagenet_train_predict_partial=True)
        predictions = model.predict(input_fn, predict_keys='classes')

        # Loop through predictions and compute accuracy.
        correct = np.zeros(len(labels))
        if self.params['dataset'] == 'imagenet':
            # For imagenet, compute top-5 accuracy.
            probabilities = np.zeros((len(labels)), 1000)
            for i, p in enumerate(predictions):
                top_5_indices = np.argsort(-p['predictions'])[:5]
                correct[i] = labels[i] in top_5_indices
        else:
            # For cifar, compute top-1 accuracy.
            for i, p in enumerate(predictions):
                correct[i] = labels[i] == p['classes']

        accuracy = np.mean(correct)

        return accuracy
