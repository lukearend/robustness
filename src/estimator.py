"""A wrapper class for the model."""

import os
import time

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

    def activations(self, data_dir='/tmp', split='train', num_gpus=1):
        """Extract activations to a dataset.

        Args:
            data_dir: directory in which to look for validation data.
            split: one of 'train' or 'validation'.
            num_gpus: number of GPUs to use.
        """
        # Configure and build the model.
        model_fn = estimator_fns.get_model_fn(num_gpus)

        config = tf.estimator.RunConfig().replace(
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        imagenet_train_predict_shuffle_seed = int(time.time())

        # # First, loop through the dataset and read out labels.
        # _, label_batch = estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
        #                                         data_dir,
        #                                         self.params,
        #                                         reading_labels=True,
        #                                         predict_split=split,
        #                                         imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed,
        #                                         imagenet_train_predict_partial=True)
        # labels = None
        # with tf.Session() as sess:
        #     while True:
        #         try:
        #             labels_tmp = sess.run(label_batch)
        #             if labels is None:
        #                 labels = labels_tmp
        #             else:
        #                 labels = np.append(labels, labels_tmp, axis=0)
        #         except tf.errors.OutOfRangeError:
        #             break

        # Extract activations.
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
        predictions = model.predict(input_fn,
                                    predict_keys=['classes', 'activations'])

        print('predict step completed')
        print(predictions)

        # Loop through predictions and store them in a numpy array.
        predicted_labels = np.zeros(np.shape(labels))
        for i, p in enumerate(predictions):
            predicted_labels[i] = p['classes']
            if i == 0:
                num_layers = len(p['activations'])
                activations_out = list(range(num_layers))
                labels_out = list(range(num_layers))

            for layer in range(num_layers):
                ###########################################
                # FIGURE OUT HOW TO RESHAPE THESE PROPERLY.
                layer_activations = np.array(p['activations'][layer])
                layer_labels = np.array(labels[i])
                ###########################################

                if i == 0:
                    activations_out[layer] = layer_activations
                    labels_out[layer] = layer_labels
                else:
                    activations_out[layer] = np.append(activations_out[layer], layer_activations, axis=0)
                    labels_out[layer] = np.append(labels_out[layer], layer_labels, axis=0)

        accuracy = np.mean(np.equals(labels, predicted_labels))

        return activations, labels, accuracy

    def robustness(self,
        perturbation_type,
        perturbation_amount,
        kill_mask,
        data_dir='/tmp',
        split='train',
        num_gpus=1):
        """Test robustness of a model.

        Args:
            perturbation_type, perturbation_amount, kill_mask:
                settings for testing robustness.
            data_dir: directory in which to look for validation data.
            split: one of 'train' or 'validation'.
            num_gpus: number of GPUs to use.
        """
        # Configure and build the model.
        model_fn = estimator_fns.get_model_fn(
            num_gpus,
            test_robustness=True,
            perturbation_type=perturbation_type,
            perturbation_amount=perturbation_amount,
            kill_mask=kill_mask)

        config = tf.estimator.RunConfig().replace(
            tf_random_seed=self.tf_random_seed,
            session_config=self.session_config)

        imagenet_train_predict_shuffle_seed = int(time.time())

        # First, loop through the dataset and read out labels.
        _, label_batch = estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
                                                data_dir,
                                                self.params,
                                                reading_labels=True,
                                                predict_split=split,
                                                imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed)
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
        model = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=config,
                                       params=self.params)
        input_fn = lambda: estimator_fns.input_fn(tf.estimator.ModeKeys.PREDICT,
                                               data_dir,
                                               self.params,
                                               num_gpus=num_gpus,
                                               predict_split=split,
                                               imagenet_train_predict_shuffle_seed=imagenet_train_predict_shuffle_seed)
        predictions = model.predict(input_fn,
                                    predict_keys='classes')

        # Loop through predictions and store them in a numpy array.
        predicted_labels = np.array([p['classes'] for p in predictions])

        accuracy = np.mean(np.equals(labels, predicted_labels))

        return accuracy
