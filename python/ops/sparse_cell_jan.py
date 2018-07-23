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


@tf_export("nn.rnn_cell.JanSparseCell")
class JanSparseCell(RNNCell):
    """Jan's sparse cell.
    """

    def __init__(self, num_spatial, num_units, num_output,
                 initializer=None,
                 name=None):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.

          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(JanSparseCell, self).__init__(_reuse=False, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._initializer = initializer
        self._output_size = num_units[-1]
        self._num_spatial = num_spatial
        self._num_output = num_output

        self._mms = []
        assert type(num_units) == list

        for x in num_units:
            assert type(x) == int
            self._mms.append(tf.layers.Dense(units=x*num_output, kernel_initializer=initializer))


    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        inputs_shape = self._num_spatial
        for i in range(len(self._num_units)):
            self._mms[i].build(inputs_shape)
            inputs_shape = self._num_units[i] * self._num_output

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

        output = inputs[:, 0:self._num_spatial]
        for i in range(len(self._num_units)):
            output = self._mms[i].call(output)

        num_input_features = int(output.shape.as_list()[1] / self._num_output)

        output = tf.reshape(output, (-1, num_input_features, self._num_output))
        inputs = tf.expand_dims(inputs[:, self._num_spatial:], axis=2)

        output = tf.reduce_sum(tf.multiply(output, inputs), axis=1)

        return output, tf.zeros_like(state)
