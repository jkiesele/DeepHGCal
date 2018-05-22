
from models.sparse_conv import SparseConv
from ops.sparse_conv import sparse_conv_2, sparse_max_pool
from ops.neighbors import indexing_tensor
import tensorflow as tf

class SparseConv2(SparseConv):

    def __init__(self, n_space, n_all, n_max_neighbors, batch_size, max_entries, num_classes, learning_rate=0.0001):
        super(SparseConv2, self).__init__(n_space, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                                         learning_rate)

    def _find_logits(self):

        indexing_tensor_1 = indexing_tensor(self._placeholder_space_features, 10)

        layer_1_out, layer_1_out_spatial = sparse_conv_2(self._placeholder_space_features,
                                                         self._placeholder_all_features, indexing_tensor_1, 15)
        layer_2_out, layer_2_out_spatial = sparse_conv_2(layer_1_out_spatial, layer_1_out, indexing_tensor_1, 20)

        # How to apply pooling:
        pool_2_out, pool_2_out_spatial, pool_2_spatial_origin =\
            sparse_max_pool(layer_2_out, 1000, [layer_2_out, layer_1_out_spatial, self._placeholder_space_features])
        indexing_tensor_2 = indexing_tensor(pool_2_spatial_origin, 10)

        layer_3_out, layer_3_out_spatial = sparse_conv_2(pool_2_out, pool_2_out_spatial, indexing_tensor_2, 25)
        layer_4_out, layer_4_out_spatial = sparse_conv_2(layer_3_out_spatial, layer_3_out, indexing_tensor_2, 30)
        layer_5_out, layer_5_out_spatial = sparse_conv_2(layer_4_out_spatial, layer_4_out, indexing_tensor_2, 35)
        layer_6_out, layer_6_out_spatial = sparse_conv_2(layer_5_out_spatial, layer_5_out, indexing_tensor_2, 40)

        entries_out = layer_6_out.get_shape().as_list()[1]

        squeezed_num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)
        mask = tf.cast(tf.expand_dims(tf.sequence_mask(squeezed_num_entries, maxlen=entries_out), axis=2),
                       tf.float32)

        nonzeros = tf.count_nonzero(mask, axis=1, dtype=tf.float32)

        # jsut safety measure - should not be necessary once done
        flattened_features = tf.clip_by_value(tf.reduce_sum(layer_6_out * mask, axis=1) / nonzeros,
                                              clip_value_min=-1e9,
                                              clip_value_max=1e9)

        fc_1 = tf.layers.dense(flattened_features, units=100, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_2 = tf.layers.dense(fc_1, units=100, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))

        return fc_3
