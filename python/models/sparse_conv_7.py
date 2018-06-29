import tensorflow as tf
from models.sparse_conv import SparseConv
from ops.sparse_conv import *
from models.switch_model import SwitchModel


class SparseConv7(SparseConv, SwitchModel):

    def __init__(self, n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries, num_classes,
                 learning_rate=0.0001):
        super(SparseConv7, self).__init__(n_space, n_space_local, n_all, n_max_neighbors, batch_size, max_entries,
                                          num_classes, learning_rate)
        self.weight_weights = []
        self.optimizer_switches = 2


    def get_place_holder_switch_control(self):
        return self._placeholder_switch_control


    def _find_logits(self):
        # # nl_all = tf.layers.dense(tf.scalar_mul(0.001, self._placeholder_all_features), units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        # # nl_all = tf.layers.dense(nl_all, units=8, activation=tf.nn.relu)
        #
        # # TODO: Remove it later after regenerating the data, this only picks energy (or do something similar)
        # net = self._placeholder_all_features
        # net = tf.concat((net, self._placeholder_space_features_local), axis=2)

        self._placeholder_switch_control = tf.placeholder(dtype=tf.int64, shape=[self.optimizer_switches])
        _input = construct_sparse_io_dict(self._placeholder_all_features, self._placeholder_space_features, self._placeholder_space_features_local,
                                          tf.squeeze(self._placeholder_num_entries))

        net = _input

        net = sparse_conv_make_neighbors(_input, num_neighbors=36, output_all=10, control_switches=self._placeholder_switch_control)
        net = sparse_conv_make_neighbors(net, num_neighbors=36, output_all=12, control_switches=self._placeholder_switch_control)
        net = sparse_conv_make_neighbors(net, num_neighbors=36, output_all=12, control_switches=self._placeholder_switch_control)
        net = sparse_max_pool(net, 600)
        net = sparse_conv_make_neighbors(net, num_neighbors=36, output_all=14, control_switches=self._placeholder_switch_control)
        net = sparse_max_pool(net, 150)
        net = sparse_conv_make_neighbors(net, num_neighbors=36, output_all=14, control_switches=self._placeholder_switch_control)
        net = sparse_max_pool(net, 40)
        flattened_features = sparse_merge_flat(net, combine_three=False)

        self._graph_temp = flattened_features[:,0:10]

        fc_1 = tf.layers.dense(flattened_features, units=30, activation=tf.nn.relu)
        fc_2 = tf.layers.dense(fc_1, units=30, activation=tf.nn.relu)
        fc_3 = tf.layers.dense(fc_2, units=self.num_classes, activation=None)

        return fc_3

    def get_variable_scope(self):
        return 'sparse_conv_v4'
