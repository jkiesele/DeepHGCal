import tensorflow as tf
from models.sparse_conv_clustering_base import SparseConvClusteringBase
from ops.sparse_conv import *
from models.switch_model import SwitchModel


class SparseConvClusteringSpatialMinLoss(SparseConvClusteringBase):

    def __init__(self, n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries, learning_rate=0.0001):
        super(SparseConvClusteringSpatialMinLoss, self).__init__(n_space, n_space_local, n_others, n_target_dim, batch_size, max_entries,
                                          learning_rate)
        self.weight_weights = []

    def _get_loss(self):
        assert self._graph_output.shape[2] == 2

        num_entries = tf.squeeze(self._placeholder_num_entries, axis=1)

        diff_sq_1 = (self._graph_output - self._placeholder_targets) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32)
        diff_sq_1 = tf.reduce_sum(diff_sq_1, axis=[-1, -2])
        loss_unreduced_1 = (diff_sq_1 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        diff_sq_2 = (self._graph_output - (1-self._placeholder_targets)) ** 2 * tf.cast(
            tf.sequence_mask(num_entries, maxlen=self.max_entries)[:, :,
            tf.newaxis], tf.float32)
        diff_sq_2 = tf.reduce_sum(diff_sq_2, axis=[-1, -2])
        loss_unreduced_2 = (diff_sq_2 / tf.cast(num_entries, tf.float32)) * tf.cast(
            num_entries != 0, tf.float32)

        return tf.reduce_mean(tf.minimum(loss_unreduced_1, loss_unreduced_2))


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

        net = sparse_conv_make_neighbors(_input, num_neighbors=18, output_all=15, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=15, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=15, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=30, spatial_degree_non_linearity=3, propagrate_ahead=True)
        net = sparse_conv_make_neighbors(net, num_neighbors=18, output_all=2, n_transformed_spatial_features=3, propagrate_ahead=True)

        output = net['all_features'] * tf.cast(tf.sequence_mask(tf.squeeze(self._placeholder_num_entries, axis=1), maxlen=self.max_entries)[:,:,tf.newaxis], tf.float32)
        output = tf.nn.softmax(output)

        self._graph_temp = output

        return output

    def get_variable_scope(self):
        return 'sparse_conv_clustering_spatial1'
