from ops.sparse_conv import *
from ops.neighbors import *
import unittest
import numpy as np


class SparseConvT(unittest.TestCase):
    def test_mem_1(self):
        color_like_features = tf.placeholder(shape=[100, 3000, 5], dtype=tf.float32)
        spatial_features_global = tf.placeholder(shape=[100, 3000, 3], dtype=tf.float32)
        spatial_features_local = tf.placeholder(shape=[100, 3000, 2], dtype=tf.float32)
        num_entries = tf.placeholder(shape=[100], dtype=tf.int64)

        net = construct_sparse_io_dict(color_like_features, spatial_features_global, spatial_features_local, num_entries)
        net = sparse_conv_loop(net, num_neighbors=27,  num_filters=32,  space_depth=2, space_relu=4, space_gauss=3)
        net = sparse_conv_loop(net, num_neighbors=27,  num_filters=32,  space_depth=2, space_relu=4, space_gauss=3)
        net = sparse_conv_loop(net, num_neighbors=27,  num_filters=32,  space_depth=2, space_relu=4, space_gauss=3)
        net = sparse_conv_loop(net, num_neighbors=27,  num_filters=32,  space_depth=2, space_relu=4, space_gauss=3)

        net = sparse_merge_flat(net)
        optim = tf.train.AdamOptimizer().minimize(loss=tf.nn.l2_loss(tf.reduce_sum(net)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                feeds = {color_like_features: np.random.randn(100,3000,5),
                         spatial_features_global: np.random.randn(100,3000,3),
                         spatial_features_local: np.random.randn(100,3000,2),
                         num_entries: np.random.randint(0,3000, 100)}
                # result = sess.run([net, optim], feed_dict=feeds)
                result = sess.run(net, feed_dict=feeds)
                print("Hello, world ", i)


if __name__ == '__main__':
    unittest.main()