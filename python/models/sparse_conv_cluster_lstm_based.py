import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from models.sparse_conv_cluster_spatial_2_min_loss import SparseConvClusteringSpatialMinLoss
from ops.sparse_conv import *
from models.switch_model import SwitchModel
from ops.sparse_conv_rec import *


class SparseConvClusterLstmBased(SparseConvClusteringSpatialMinLoss):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(SparseConvClusterLstmBased, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries,
                                          learning_rate)
        self.weight_weights = []



    def _compute_output(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        # net = self._placeholder_all_features
        # net = tf.concat((net, self._placeholder_space_features_local), axis=2)

        # TODO: Will cause problems with batch size of 1

        _input = construct_sparse_io_dict(self._placeholder_other_features, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = sparse_conv_rec(_input, num_neighbors=18, output_all=35)
        net = sparse_conv_rec(net, num_neighbors=18, output_all=43)
        net = sparse_conv_rec(net, num_neighbors=18, output_all=43)
        net = sparse_conv_rec(net, num_neighbors=18, output_all=43)
        net = sparse_conv_rec(net, num_neighbors=18, output_all=43)
        net = sparse_conv_rec(net, num_neighbors=18, output_all=43)
        net = sparse_conv_rec(net, num_neighbors=18, output_all=3)

        output = net['all_features'] * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        output = tf.nn.softmax(output)

        self._graph_temp = output

        return output

    def get_variable_scope(self):
        return 'sparse_conv_clustering_lstm'