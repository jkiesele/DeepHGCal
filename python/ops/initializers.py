
import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import random_ops

class NoisyEyeInitializer(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, low=-0.01, up=0.01, seed=None, dtype=tf.float32):
    self.low = low
    self.high = up
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    assert len(shape) == 2
    return tf.eye(shape[0], shape[1]) + random_ops.random_uniform(
        shape, self.low, self.high, dtype, seed=self.seed)

  def get_config(self):
    return {
        "low": self.low,
        "high": self.high,
        "seed": self.seed,
        "dtype": self.dtype.name
    }



