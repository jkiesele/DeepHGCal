from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

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
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import RNNCell, LSTMStateTuple
import tensorflow as tf
from ops.sparse_conv import gauss_activation


@tf_export("nn.rnn_cell.JanSparseCell")
class JanSparseCell2(RNNCell):
    """Jan's sparse cell.
    """

    def __init__(self, num_spatial, num_output=10, depth=1, relu_units=5, gauss_units=5,
                 initializer=None,
                 name=None):
        """Initialize the parameters for an LSTM cell.

        Args:
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.

          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(JanSparseCell2, self).__init__(_reuse=False, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._initializer = initializer
        self._num_spatial = num_spatial
        self._num_output = num_output
        self._output_size = num_output

        self._relu_units = relu_units
        self._gauss_units = gauss_units
        self._depth = depth

        self._first = tf.layers.Dense(units=num_output*10)



    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        # inputs_shape = self._num_spatial
        self._first.build(self._num_spatial)

        n_non_spatial = inputs_shape.as_list()[1] - self._num_spatial


        # TODO: Initialize weights_kernel_relu
        # TODO: Initialize weights_bias_relu

        # TODO: Initialize weights_kernel_gauss
        # TODO: Initialize weights_bias_gaus

        # TODO: Initialize weights_kernel_output
        # TODO: Initialize weights_bias_output

        self.weights_kernel_relu = list()
        self.weights_bias_relu = list()
        self.weights_kernel_gauss = list()
        self.weights_bias_gauss = list()
        in_shape = self._relu_units + self._gauss_units
        for i in range(self._depth):
            self.weights_kernel_relu.append(self.add_variable('kernel_relu'+str(i),
                                                           initializer=self._initializer,
                                                           shape=[1, self._num_output, self._relu_units,
                                                                  in_shape],
                                                           dtype=tf.float32))

            self.weights_bias_relu.append(self.add_variable('bias_relu'+str(i),
                                                         initializer=tf.initializers.zeros,
                                                         shape=[1, self._num_output, self._relu_units],
                                                         dtype=tf.float32))

            self.weights_kernel_gauss.append(self.add_variable('kernel_gauss'+str(i),
                                                           initializer=self._initializer,
                                                           shape=[1, self._num_output, self._gauss_units,
                                                                  in_shape],
                                                           dtype=tf.float32))
            self.weights_bias_gauss.append(self.add_variable('bias_gauss'+str(i),
                                                         initializer=tf.initializers.zeros,
                                                         shape=[1, self._num_output, self._gauss_units],
                                                         dtype=tf.float32))
            in_shape = self._gauss_units + self._relu_units


        self.weights_kernel_output = self.add_variable('kernel_output',
                                                            initializer=self._initializer,
                                                            shape=[1, self._num_output, n_non_spatial, 1],
                                                            dtype=tf.float32)
        self.weights_bias_output = self.add_variable('bias_output',
                                                            initializer=tf.initializers.zeros,
                                                            shape=[1, self._num_output, n_non_spatial],
                                                            dtype=tf.float32)



        self.weights_kernel_output_bias = self.add_variable('kernel_output_bias',
                                                            initializer=self._initializer,
                                                            shape=[1, self._num_output, 1, 1],
                                                            dtype=tf.float32)
        self.weights_bias_output_bias = self.add_variable('bias_output_bias',
                                                            initializer=tf.initializers.zeros,
                                                            shape=[1, self._num_output, 1],
                                                            dtype=tf.float32)
        self.built = True

    def call(self, inputs, state):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, `[batch, num_units].
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch, output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """

        inputs_shape = inputs.get_shape().as_list()
        n_batch = inputs_shape[0]
        n_total_input = inputs_shape[1]
        n_non_spatial_input = n_total_input - self._num_spatial

        output = inputs[:, 0:self._num_spatial]

        non_spatial = inputs[:, self._num_spatial:]

        weight_values = self._first.call(output)
        weight_values = tf.reshape(weight_values, [n_batch, self._num_output, -1])

        for i in range(self._depth):
            relu_output = tf.reduce_sum(tf.multiply(self.weights_kernel_relu[i], tf.expand_dims(weight_values, axis=-2)), axis=-1)
            relu_output = tf.add(self.weights_bias_relu[i], relu_output)
            relu_output = tf.nn.relu(relu_output)

            gauss_output = tf.reduce_sum(tf.multiply(self.weights_kernel_gauss[i], tf.expand_dims(weight_values, axis=-2)), axis=-1)
            gauss_output = tf.add(self.weights_bias_gauss[i], gauss_output)
            gauss_output = gauss_activation(gauss_output)

            weight_values = tf.concat((relu_output, gauss_output), axis=-1)

        weight_values_kernel = tf.reduce_sum(tf.multiply(self.weights_kernel_output, tf.expand_dims(weight_values, axis=-2)),  axis=-1)
        weight_values_kernel = tf.add(self.weights_bias_output, weight_values_kernel)

        weight_values_bias = tf.reduce_sum(tf.multiply(self.weights_kernel_output_bias, tf.expand_dims(weight_values, axis=-2)),axis=-1)

        weight_values_bias = tf.add(self.weights_bias_output_bias, weight_values_bias)

        non_spatial = tf.expand_dims(non_spatial, axis=-2)
        output = tf.reduce_sum(tf.multiply(non_spatial, weight_values_kernel), axis=-1)

        output = tf.add(output, tf.squeeze(weight_values_bias, axis=-1))

        return output, tf.zeros_like(state)
