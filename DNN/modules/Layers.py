## empty
import tensorflow as tf
from keras.engine.topology import Layer
import keras.backend as K
from keras.engine import InputSpec
import copy
import numpy as np


class ReshapeBatch(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ReshapeBatch, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReshapeBatch, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return tf.reshape(x, self.output_dim)

    def compute_output_shape(self, input_shape):
        return self.output_dim





class PermuteBatch(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful for e.g. connecting RNNs and convnets together.

    # Example

    ```python
        model = Sequential()
        model.add(Permute((2, 1), input_shape=(10, 64)))
        # now: model.output_shape == (None, 64, 10)
        # note: `None` is the batch dimension
    ```

    # Arguments
        dims: Tuple of integers. Permutation pattern, includes the
            samples dimension. Indexing starts at 0.
            For instance, `(1, 0)` permutes the first and second dimension
            of the input.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, includes the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.
    """

    def __init__(self, dims, **kwargs):
        super(PermuteBatch, self).__init__(**kwargs)
        self.dims = tuple(dims)
        self.input_spec = InputSpec(ndim=len(self.dims))

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i] = target_dim
        return tuple(output_shape)

    def call(self, inputs):
        return K.permute_dimensions(inputs, self.dims)

    def get_config(self):
        config = {'dims': self.dims}
        base_config = super(PermuteBatch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class ReshapeBatch(Layer):
    """Reshapes an output to a certain shape.

    # Arguments
        target_shape: target shape. Tuple of integers.
            Includes the batch axis.

    # Input shape
        Arbitrary, although all dimensions in the input shaped must be fixed.
        Use the keyword argument `input_shape`
        (tuple of integers, includes the batch axis)
        when using this layer as the first layer in a model.

    # Output shape
        `target_shape`

    """

    def __init__(self, target_shape, **kwargs):
        super(ReshapeBatch, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        # input shape known? then we can compute the output shape
        return self.target_shape

    def call(self, inputs):
        return K.reshape(inputs, self.target_shape)

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(ReshapeBatch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
