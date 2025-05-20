from __future__ import print_function
from collections import deque
from datetime import datetime
import logging
import os
import pprint as pp
import time

import numpy as np
import tensorflow as tf

from tf_utils import shape

# Enable TensorFlow 2.x compatibility
tf.compat.v1.disable_eager_execution()

class TFBaseModel(object):

    """Interface containing some boilerplate code for training tensorflow models.

    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.

    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_steps:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
    """

    def __init__(
        self,
        reader=None,
        batch_sizes=[128],
        num_training_steps=20000,
        learning_rates=[.01],
        beta1_decays=[.99],
        optimizer='adam',
        grad_clip=5,
        regularization_constant=0.0,
        keep_prob=1.0,
        patiences=[3000],
        warm_start_init_step=0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=100,
        log_interval=20,
        logging_level=logging.INFO,
        loss_averaging_window=100,
        validation_batch_size=64,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
    ):

        assert len(batch_sizes) == len(learning_rates) == len(patiences)
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.beta1_decays = beta1_decays
        self.patiences = patiences
        self.num_restarts = len(batch_sizes) - 1
        self.restart_idx = 0
        self.update_train_params()

        self.reader = reader
        self.num_training_steps = num_training_steps
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.keep_prob_scalar = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.loss_averaging_window = loss_averaging_window
        self.validation_batch_size = validation_batch_size

        self.log_dir = log_dir
        self.logging_level = logging_level
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'

        self.init_logging(self.log_dir)
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))

        self.graph = self.build_graph()
        self.session = tf.compat.v1.Session(graph=self.graph)
        logging.info('built graph')

    def update_train_params(self):
        self.batch_size = self.batch_sizes[self.restart_idx]
        self.learning_rate = self.learning_rates[self.restart_idx]
        self.beta1_decay = self.beta1_decays[self.restart_idx]
        self.early_stopping_steps = self.patiences[self.restart_idx]

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    def fit(self):
        with self.session.as_default():

            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                self.session.run(self.init)
                step = 0

            train_generator = self.reader.train_batch_generator(self.batch_size)
            val_generator = self.reader.val_batch_generator(self.validation_batch_size)

            train_loss_history = deque(maxlen=self.loss_averaging_window)
            val_loss_history = deque(maxlen=self.loss_averaging_window)
            train_time_history = deque(maxlen=self.loss_averaging_window)
            val_time_history = deque(maxlen=self.loss_averaging_window)
            if not hasattr(self, 'metrics'):
                self.metrics = {}

            metric_histories = {
                metric_name: deque(maxlen=self.loss_averaging_window) for metric_name in self.metrics
            }
            best_validation_loss, best_validation_tstep = float('inf'), 0

            while step < self.num_training_steps:

                # validation evaluation
                val_start = time.time()
                val_batch_df = next(val_generator)
                val_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in val_batch_df.items() if hasattr(self, placeholder_name)
                }

                val_feed_dict.update({self.learning_rate_var: self.learning_rate, self.beta1_decay_var: self.beta1_decay})
                if hasattr(self, 'keep_prob'):
                    val_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    val_feed_dict.update({self.is_training: False})

                results = self.session.run(
                    fetches=[self.loss] + list(self.metrics.values()),
                    feed_dict=val_feed_dict
                )
                val_loss = results[0]
                val_metrics = results[1:] if len(results) > 1 else []
                val_metrics = dict(zip(self.metrics.keys(), val_metrics))
                val_loss_history.append(val_loss)
                val_time_history.append(time.time() - val_start)
                for key in val_metrics:
                    metric_histories[key].append(val_metrics[key])

                if hasattr(self, 'monitor_tensors'):
                    for name, tensor in self.monitor_tensors.items():
                        [np_val] = self.session.run([tensor], feed_dict=val_feed_dict)
                        print(name)
                        print('min', np_val.min())
                        print('max', np_val.max())
                        print('mean', np_val.mean())
                        print('std', np_val.std())
                        print('nans', np.isnan(np_val).sum())
                        print()
                    print()
                    print()

                # train step
                train_start = time.time()
                train_batch_df = next(train_generator)
                train_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in train_batch_df.items() if hasattr(self, placeholder_name)
                }

                train_feed_dict.update({self.learning_rate_var: self.learning_rate, self.beta1_decay_var: self.beta1_decay})
                if hasattr(self, 'keep_prob'):
                    train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
                if hasattr(self, 'is_training'):
                    train_feed_dict.update({self.is_training: True})

                results = self.session.run(
                    fetches=[self.train_op, self.loss] + list(self.metrics.values()),
                    feed_dict=train_feed_dict
                )
                train_loss = results[1]
                train_metrics = results[2:] if len(results) > 2 else []
                train_metrics = dict(zip(self.metrics.keys(), train_metrics))
                train_loss_history.append(train_loss)
                train_time_history.append(time.time() - train_start)
                for key in train_metrics:
                    metric_histories[key].append(train_metrics[key])

                if step % self.log_interval == 0:
                    logging.info(
                        'step {}: train loss {:.4f}, val loss {:.4f}, train time {:.4f}, val time {:.4f}'.format(
                            step,
                            np.mean(train_loss_history),
                            np.mean(val_loss_history),
                            np.mean(train_time_history),
                            np.mean(val_time_history)
                        )
                    )
                    for metric_name, metric_history in metric_histories.items():
                        logging.info('{}: {:.4f}'.format(metric_name, np.mean(metric_history)))

                if step % self.min_steps_to_checkpoint == 0:
                    self.save(step)
                    if self.enable_parameter_averaging:
                        self.save(step, averaged=True)

                if np.mean(val_loss_history) < best_validation_loss:
                    best_validation_loss = np.mean(val_loss_history)
                    best_validation_tstep = step
                    self.save(step, is_best=True)
                    if self.enable_parameter_averaging:
                        self.save(step, averaged=True, is_best=True)

                if step - best_validation_tstep > self.early_stopping_steps:
                    if self.restart_idx < self.num_restarts:
                        self.restart_idx += 1
                        self.update_train_params()
                        self.save(step, is_best=True)
                        if self.enable_parameter_averaging:
                            self.save(step, averaged=True, is_best=True)
                        self.restore(step, averaged=True)
                        best_validation_loss = float('inf')
                        best_validation_tstep = step
                    else:
                        break

                step += 1

            if step <= self.min_steps_to_checkpoint:
                best_validation_tstep = step
                self.save(step)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

            logging.info('num_training_steps reached - ending training')

    def predict(self, chunk_size=256):
        if not os.path.isdir(self.prediction_dir):
            os.makedirs(self.prediction_dir)

        if hasattr(self, 'prediction_tensors'):
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}

            test_generator = self.reader.test_batch_generator(chunk_size)
            for i, test_batch_df in enumerate(test_generator):
                if i % 10 == 0:
                    print(i*len(test_batch_df))

                test_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in test_batch_df.items() if hasattr(self, placeholder_name)
                }
                if hasattr(self, 'keep_prob'):
                    test_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    test_feed_dict.update({self.is_training: False})

                tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                np_tensors = self.session.run(
                    fetches=tf_tensors,
                    feed_dict=test_feed_dict
                )
                for tensor_name, tensor in zip(tensor_names, np_tensors):
                    prediction_dict[tensor_name].append(tensor)

            for tensor_name, tensor in prediction_dict.items():
                np_tensor = np.concatenate(tensor, 0)
                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

        if hasattr(self, 'parameter_tensors'):
            for tensor_name, tensor in self.parameter_tensors.items():
                np_tensor = tensor.eval(self.session)

                save_file = os.path.join(self.prediction_dir, '{}.npy'.format(tensor_name))
                logging.info('saving {} with shape {} to {}'.format(tensor_name, np_tensor.shape, save_file))
                np.save(save_file, np_tensor)

    def save(self, step, averaged=False, is_best=False):
        if averaged:
            saver = tf.compat.v1.train.Saver(self.avg_dict)
            checkpoint_dir = self.checkpoint_dir_averaged
        else:
            saver = tf.compat.v1.train.Saver()
            checkpoint_dir = self.checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if is_best:
            saver.save(self.session, os.path.join(checkpoint_dir, 'best.ckpt'))
        else:
            saver.save(self.session, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=step)

    def restore(self, step=None, averaged=False):
        if averaged:
            saver = tf.compat.v1.train.Saver(self.avg_dict)
            checkpoint_dir = self.checkpoint_dir_averaged
        else:
            saver = tf.compat.v1.train.Saver()
            checkpoint_dir = self.checkpoint_dir

        if step is None:
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt-{}'.format(step))

        saver.restore(self.session, checkpoint_path)

    def init_logging(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logging.basicConfig(
            level=self.logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )

    def update_parameters(self, loss):
        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([tf.reduce_sum(tf.square(param)) for param in tf.compat.v1.trainable_variables()])
            loss = loss + self.regularization_constant * l2_norm

        optimizer = self.get_optimizer(self.learning_rate_var, self.beta1_decay_var)
        grads, params = zip(*optimizer.compute_gradients(loss))
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        return optimizer.apply_gradients(zip(grads, params))

    def get_optimizer(self, learning_rate, beta1_decay):
        if self.optimizer == 'adam':
            return tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=beta1_decay)
        elif self.optimizer == 'rms':
            return tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay=beta1_decay)
        elif self.optimizer == 'sgd':
            return tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer {} not recognized'.format(self.optimizer))

    def build_graph(self):
        """Build the computational graph for training."""
        with tf.Graph().as_default() as graph:
            # Create placeholders
            self.learning_rate_var = tf.compat.v1.placeholder(tf.float32, shape=[], name='learning_rate')
            self.beta1_decay_var = tf.compat.v1.placeholder(tf.float32, shape=[], name='beta1_decay')
            
            # Create the model and calculate loss
            self.loss = self.calculate_loss()
            
            # Create optimizer and training op
            self.optimizer = self.get_optimizer(self.learning_rate_var, self.beta1_decay_var)
            self.train_op = self.optimizer.minimize(self.loss)
            
            # Initialize variables
            self.init = tf.compat.v1.global_variables_initializer()
            
            # Create saver
            self.saver = tf.compat.v1.train.Saver()
            
            return graph
