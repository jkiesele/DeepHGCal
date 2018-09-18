from ops.sparse_conv_2 import *
from ops.neighbors import *
import unittest
import numpy as np


class SparseConvT(unittest.TestCase):
    def test_syntax(self):
        color_like_features = tf.placeholder(shape=[100, 3000, 5], dtype=tf.float32)
        spatial_features_global = tf.placeholder(shape=[100, 3000, 3], dtype=tf.float32)
        spatial_features_local = tf.placeholder(shape=[100, 3000, 2], dtype=tf.float32)
        num_entries = tf.placeholder(shape=[100], dtype=tf.int64)

        dict_input = construct_sparse_io_dict(color_like_features, spatial_features_global, spatial_features_local, num_entries)
        output = sparse_conv(dict_input, 9, 10, n_prespace_conditions=5)

    def test_n_range(self):
        dims = [3, 3]

        should_be_equal_to = np.array([[[0,0], [0,1], [0,2]], [[1,0], [1,1], [1,2]], [[2,0], [2,1], [2,2]]])

        op = n_range_tensor(dims)

        with tf.Session() as sess:
            result = sess.run(op)
            assert (np.allclose(result, should_be_equal_to))

    def test_sorting(self):
        random_tensor = tf.placeholder(shape=[10, 10], dtype=tf.float32)

        sort_it = sort_last_dim_tensor(random_tensor)
        sort_it = tf.gather_nd(random_tensor, sort_it)
        with tf.Session() as sess:
            input_rand = np.random.random((10,10))
            result = sess.run(sort_it, feed_dict={random_tensor:input_rand})
            print(result)


if __name__ == '__main__':
    unittest.main()