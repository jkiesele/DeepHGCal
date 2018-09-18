import tensorflow as tf
from tensorflow.python.layers.base import Layer
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class GRUCell(Layer):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(GRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    _WEIGHTS_VARIABLE_NAME='WEIGHT_1_1'
    _BIAS_VARIABLE_NAME='BIAS_1_1'

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=tf.float32)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=tf.float32)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h




def message_parsing_network(x, h, time, num_classes):
    num_hidden = h.shape[2]
    num_vertices = h.shape[1]
    num_input = x.shape[2]

    adjacency_matrix_forward = tf.layers.Dense(1, activation=None)
    adjacency_matrix_forward.build(input_shape=num_hidden)
    gru_cell = GRUCell(num_hidden)

    gru_cell.build(inputs_shape=[h.shape[0]*h.shape[1], num_input+num_hidden])

    for i in range(time):
        # Adjacency matrix computation
        A = adjacency_matrix_forward.call(
            tf.add(
                tf.tile(tf.expand_dims(h, axis=2), multiples=[1, 1, num_vertices, 1]),
                tf.tile(tf.expand_dims(h, axis=1), multiples=[1, num_vertices, 1, 1])
            )
        )
        A = tf.nn.softmax(tf.squeeze(A, axis=3))
        # A is now of shape [B,V,V]
        m = tf.tanh(tf.reduce_sum(
            tf.expand_dims(A, axis=-1) * tf.tile(tf.expand_dims(h, axis=1), multiples=[1, num_vertices, 1, 1]), axis=2))

        gru_input = tf.reshape(tf.concat([m, x], axis=2), (-1, num_hidden+num_input))
        gru_input_h = tf.reshape(h, shape=(-1, num_hidden))

        h,_ = gru_cell.call(gru_input, gru_input_h)
        h = tf.reshape(h, shape=(-1, num_vertices, num_hidden))

    return tf.layers.dense(tf.expand_dims(tf.reduce_sum(tf.reduce_sum(A, axis=-1), axis=-1), axis=-1), units=num_classes, activation=None)