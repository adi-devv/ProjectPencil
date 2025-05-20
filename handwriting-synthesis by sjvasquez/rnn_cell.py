from __future__ import print_function
import numpy as np
import tensorflow as tf

from rnn_ops import linear
from tf_utils import time_distributed_dense_layer

# Enable TensorFlow 2.x compatibility
tf.compat.v1.disable_eager_execution()

class LSTMAttentionCell(tf.keras.layers.Layer):
    def __init__(
        self,
        lstm_size,
        num_attn_mixture_components,
        attention_values,
        attention_values_lengths,
        num_output_mixture_components,
        bias,
        reuse=None
    ):
        super(LSTMAttentionCell, self).__init__()
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = attention_values
        self.attention_values_lengths = attention_values_lengths
        self.num_output_mixture_components = num_output_mixture_components
        self.bias = bias
        self._reuse = reuse

        self._num_units = lstm_size
        self._state_is_tuple = True
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_size)
        
        # Initialize variables in build method
        self.attention_matrix = None
        self.attention_bias = None
        self.output_weights = None
        self.output_bias = None

    def build(self, input_shape):
        # Create variables for attention mechanism
        with tf.compat.v1.variable_scope('attention', reuse=self._reuse):
            self.attention_matrix = self.add_weight(
                name='Matrix',
                shape=[self.num_attn_mixture_components*3, self.lstm_size],
                initializer='glorot_uniform',
                trainable=True
            )
            self.attention_bias = self.add_weight(
                name='Bias',
                shape=[self.num_attn_mixture_components*3],
                initializer='zeros',
                trainable=True
            )
        
        # Create variables for output layer
        with tf.compat.v1.variable_scope('output', reuse=self._reuse):
            self.output_weights = self.add_weight(
                name='weights',
                shape=[self.lstm_size + self.attention_values.get_shape().as_list()[-1], self.num_output_mixture_components*6 + 1],
                initializer='glorot_uniform',
                trainable=True
            )
            self.output_bias = self.add_weight(
                name='biases',
                shape=[self.num_output_mixture_components*6 + 1],
                initializer='zeros',
                trainable=True
            )
        super(LSTMAttentionCell, self).build(input_shape)

    @property
    def state_size(self):
        return tf.keras.layers.LSTMCell(self._num_units).state_size

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.
        Returns:
            tensor of shape '[batch_size x state_size]' filled with zeros.
        """
        return tf.zeros([batch_size, self._num_units], dtype=dtype), tf.zeros([batch_size, self._num_units], dtype=dtype)

    def call(self, inputs, state):
        c, h = state

        # attention
        attention = self.attention(h)
        context = tf.matmul(attention, self.attention_values)
        context = tf.squeeze(context, axis=1)  # Remove the middle dimension

        # lstm
        lstm_input = tf.concat([inputs, context], axis=1)
        lstm_output, lstm_state = self.lstm_cell(lstm_input, state)

        # output
        output = tf.concat([lstm_output, context], axis=1)
        output = tf.matmul(output, self.output_weights) + self.output_bias

        return output, lstm_state

    def attention(self, h):
        # attention parameters
        attention_params = tf.matmul(h, self.attention_matrix, transpose_b=True) + self.attention_bias
        attention_params = tf.reshape(attention_params, [-1, self.num_attn_mixture_components, 3])
        alpha, beta, kappa = tf.unstack(attention_params, axis=2)
        alpha = tf.exp(alpha)
        beta = tf.exp(beta)
        kappa = tf.exp(kappa)

        # attention
        u = tf.cumsum(kappa, axis=1)
        u = tf.expand_dims(u, 2)
        attention = tf.reduce_sum(
            alpha * tf.exp(-beta * tf.square(u - tf.expand_dims(tf.range(tf.shape(self.attention_values)[1], dtype=tf.float32), 0))),
            axis=1
        )
        attention = attention / tf.reduce_sum(attention, axis=1, keepdims=True)
        attention = tf.expand_dims(attention, 1)

        return attention
