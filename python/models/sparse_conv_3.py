
from models.sparse_conv import SparseConv
from ops.sparse_conv import sparse_conv_2, sparse_max_pool
from ops.neighbors import indexing_tensor
import tensorflow as tf

class SparseConv2(SparseConv):

    def __init__(self, n_space, n_all, n_max_neighbors, batch_size, max_entries, num_classes, learning_rate=0.0001):
        super(SparseConv2, self).__init__(n_space, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                                         learning_rate)

    def _find_logits(self):
        indexing_tensor_1 = indexing_tensor(self._placeholder_space_features, self.n_max_neighbors)
        
        
        # get right order of magnitude. prevents hughe losses in the beginning a bit
        scaled_features = tf.scalar_mul(0.001, self._placeholder_space_features) 

        layer_1_out, layer_1_out_spatial = sparse_conv_2(self._placeholder_space_features,
                                                         scaled_features, indexing_tensor_1, 25)
        
        
        layer_2_out, layer_2_out_spatial = sparse_conv_2(layer_1_out_spatial, layer_1_out, indexing_tensor_1, 25)

        # How to apply pooling:
        pool_2_out, pool_2_out_spatial, pool_2_spatial_origin =\
            sparse_max_pool(layer_2_out, 250, [layer_2_out, layer_2_out_spatial, self._placeholder_space_features])
        indexing_tensor_2 = indexing_tensor(pool_2_spatial_origin, self.n_max_neighbors)
        

        layer_3_out, layer_3_out_spatial = sparse_conv_2(pool_2_out_spatial, pool_2_out , indexing_tensor_2, 15)
        
        pool_3_out, pool_3_out_spatial, pool_3_spatial_origin =\
            sparse_max_pool(layer_2_out, 50, [layer_3_out, layer_3_out_spatial, pool_2_spatial_origin])
        indexing_tensor_3 = indexing_tensor(pool_3_spatial_origin, self.n_max_neighbors)
        
        layer_4_out = sparse_conv_2(pool_3_out_spatial, pool_3_out, indexing_tensor_3, 3,
                                                         merge_space_and_colour=True)
 
        flattened_features=tf.layers.flatten(layer_4_out)
        

        ### move the masking to the sparseConv layers directly (let them produce zeros where there is nothing)


        # jsut safety measure - should not be necessary once done
        # flattened_features = tf.clip_by_value(flattened_features,clip_value_min=-1e9,clip_value_max=1e9)
        
        fc_1 = tf.layers.dense(flattened_features, units=100, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None,
                               kernel_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width),
                               bias_initializer=tf.random_normal_initializer(mean=0., stddev=self.weight_init_width))

        return fc_3
