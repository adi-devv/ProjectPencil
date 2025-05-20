from __future__ import print_function
import tensorflow as tf

# Enable TensorFlow 2.x compatibility
tf.compat.v1.disable_eager_execution()

def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                dropout=None, scope='dense-layer', reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].
    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.
    Returns:
        Tensor of shape [batch size, output_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def time_distributed_dense_layer(inputs, output_dim, scope=None, reuse=None):
    """
    Applies a time distributed dense layer to the inputs.
    Args:
        inputs: tensor of shape [batch_size, time_steps, input_dim]
        output_dim: dimension of the output
        scope: variable scope name
        reuse: whether to reuse variables
    Returns:
        tensor of shape [batch_size, time_steps, output_dim]
    """
    with tf.compat.v1.variable_scope(scope or 'time_distributed_dense', reuse=reuse):
        input_dim = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable('weights', [input_dim, output_dim])
        biases = tf.compat.v1.get_variable('biases', [output_dim])
        outputs = tf.matmul(inputs, weights) + biases
        return outputs


def shape(tensor, dim=None):
    if dim is None:
        return tensor.get_shape().as_list()
    else:
        return tensor.get_shape().as_list()[dim]


def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())
