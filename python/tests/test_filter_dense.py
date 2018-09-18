from ops.sparse_conv_2 import *
from ops.neighbors import *
import unittest
import numpy as np
from ops.nn import FilterWiseDense
from ops.nn import *


class SparseConvT2(unittest.TestCase):
    def test_matmul(self):
        input = tf.placeholder(shape=[101, 13, 17, 21], dtype=tf.float32)
        filter = tf.placeholder(shape=[17, 21], dtype=tf.float32)

        op = tf.multiply(input, filter)
        op = tf.reduce_sum(op, axis=-1)
        print(op.shape)

    def test_filter_dense(self):
        input = tf.placeholder(shape=[101, 13, 17, 21], dtype=tf.float32)

        output = filter_wise_dense(input)

        print(output.shape)

if __name__ == '__main__':
    unittest.main()