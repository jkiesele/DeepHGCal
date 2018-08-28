import tensorflow as tf
from .neighbors import indexing_tensor, indexing_tensor_2, sort_last_dim_tensor
from ops.nn import *
import numpy as np
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import random_ops
from ops.sparse_conv import *
from ops.sparse_cell_jan_2 import JanSparseCell2


_sparse_global_lstm_index = 0
def sparse_conv_rec(sparse_dict, num_neighbors=10, output_all=15):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :param weight_init_width: TODO: Fill this
    :return: Dictionary containing output which can be made input to the next layer
    """

    global _sparse_global_lstm_index

    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    _indexing_tensor = indexing_tensor(spatial_features_global, num_neighbors)

    shape_space_features = spatial_features_global.get_shape().as_list()
    shape_space_features_local = spatial_features_local.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_indexing_tensor = _indexing_tensor.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]
    n_max_neighbors = shape_indexing_tensor[2]

    # All of these tensors should be 3-dimensional
    # TODO: Add assert for indexing_tensor shape
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_indexing_tensor) == 4

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0]
    assert shape_space_features[1] == shape_all_features[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert _indexing_tensor.dtype == tf.int64

    gathered_space_1 = tf.gather_nd(spatial_features_global, _indexing_tensor)  # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, spatial_features_global)  # [B,E,5,S]

    spatial_features_local_gathered = tf.gather_nd(spatial_features_local, _indexing_tensor)

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    gathered_flattenned_temporal = tf.reshape(gathered_all, (-1, n_max_neighbors, n_features_input_all))

    lstm_cell = tf.nn.rnn_cell.LSTMCell(output_all)
    x, state = tf.nn.dynamic_rnn(lstm_cell, gathered_flattenned_temporal,
                                 sequence_length=tf.ones(n_max_entries * n_batch, dtype=tf.int32) * n_max_neighbors,
                                 dtype=tf.float32, time_major=False, scope=('lstm_%d' % _sparse_global_lstm_index))
    _sparse_global_lstm_index += 1

    output = tf.reshape(state.h, shape=(n_batch, n_max_entries, -1))
    output = tf.layers.dense(output, units=output_all)

    return construct_sparse_io_dict(output, spatial_features_global, spatial_features_local, num_entries)



def sparse_conv_add(sparse_dict, num_neighbors=10, output_all=15):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :param weight_init_width: TODO: Fill this
    :return: Dictionary containing output which can be made input to the next layer
    """

    global _sparse_global_lstm_index

    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    _indexing_tensor = indexing_tensor(spatial_features_global, num_neighbors)

    shape_space_features = spatial_features_global.get_shape().as_list()
    shape_space_features_local = spatial_features_local.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_indexing_tensor = _indexing_tensor.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]
    n_max_neighbors = shape_indexing_tensor[2]

    # All of these tensors should be 3-dimensional
    # TODO: Add assert for indexing_tensor shape
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_indexing_tensor) == 4

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0]
    assert shape_space_features[1] == shape_all_features[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert _indexing_tensor.dtype == tf.int64

    gathered_space_1 = tf.gather_nd(spatial_features_global, _indexing_tensor)  # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, spatial_features_global)  # [B,E,5,S]

    spatial_features_local_gathered = tf.gather_nd(spatial_features_local, _indexing_tensor)

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    output = tf.layers.dense(gathered_all, output_all, activation=tf.nn.relu)
    output = tf.reduce_sum(output, axis=2)


    return construct_sparse_io_dict(output, spatial_features_global, spatial_features_local, num_entries)


def sparse_conv_jan(sparse_dict, num_neighbors=10, output_all=15):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :param weight_init_width: TODO: Fill this
    :return: Dictionary containing output which can be made input to the next layer
    """

    global _sparse_global_lstm_index

    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    _indexing_tensor = indexing_tensor(spatial_features_global, num_neighbors)

    shape_space_features = spatial_features_global.get_shape().as_list()
    shape_space_features_local = spatial_features_local.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_indexing_tensor = _indexing_tensor.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]
    n_max_neighbors = shape_indexing_tensor[2]

    # All of these tensors should be 3-dimensional
    # TODO: Add assert for indexing_tensor shape
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_indexing_tensor) == 4

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0]
    assert shape_space_features[1] == shape_all_features[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert _indexing_tensor.dtype == tf.int64

    gathered_space_1 = tf.gather_nd(spatial_features_global, _indexing_tensor)  # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, spatial_features_global)  # [B,E,5,S]

    spatial_features_local_gathered = tf.gather_nd(spatial_features_local, _indexing_tensor)

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    data_in = tf.concat((gathered_all, gathered_space_1), axis=3)
    gathered_flattenned_temporal = tf.reshape(data_in, (n_batch * n_max_entries, n_max_neighbors, -1))

    # lstm_cell = JanSparseCell2(n_features_input_space, [5, 5, n_features_input_all], num_output=5, initializer=tf.random_normal_initializer(0.002, 0.002))
    lstm_cell = JanSparseCell2(n_features_input_space,
                               initializer=tf.random_normal_initializer(0.002, 0.002))
    x, state = tf.nn.dynamic_rnn(lstm_cell, gathered_flattenned_temporal,
                                 sequence_length=tf.ones(n_max_entries * n_batch, dtype=tf.int32) * n_max_neighbors,
                                 dtype=tf.float32, time_major=False, scope=('sparse_%d' % _sparse_global_lstm_index), swap_memory=True)
    _sparse_global_lstm_index += 1

    output = tf.reduce_sum(x, axis=1)
    output = tf.reshape(output, shape=(n_batch, n_max_entries, -1))

    return construct_sparse_io_dict(output, spatial_features_global, spatial_features_local, num_entries)


def sparse_conv_jan_with_conv(sparse_dict, num_neighbors=10, output_all=15):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :param weight_init_width: TODO: Fill this
    :return: Dictionary containing output which can be made input to the next layer
    """

    global _sparse_global_lstm_index

    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    _indexing_tensor = indexing_tensor(spatial_features_global, num_neighbors)

    shape_space_features = spatial_features_global.get_shape().as_list()
    shape_space_features_local = spatial_features_local.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_indexing_tensor = _indexing_tensor.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]
    n_max_neighbors = shape_indexing_tensor[2]

    # All of these tensors should be 3-dimensional
    # TODO: Add assert for indexing_tensor shape
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_indexing_tensor) == 4

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0]
    assert shape_space_features[1] == shape_all_features[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert _indexing_tensor.dtype == tf.int64

    gathered_space_1 = tf.gather_nd(spatial_features_global, _indexing_tensor)  # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, spatial_features_global)  # [B,E,5,S]

    spatial_features_local_gathered = tf.gather_nd(spatial_features_local, _indexing_tensor)

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    expanded_space = tf.reshape(gathered_space_1, shape=(n_batch, n_max_entries * num_neighbors, -1))


    outputs = []
    for i in range(output_all):
        space_to_weights = tf.layers.conv1d(expanded_space, filters=10, kernel_size=1, activation=tf.nn.relu)
        space_to_weights = tf.layers.conv1d(space_to_weights, filters=10, kernel_size=1, activation=tf.nn.relu)

        space_to_weights_kernel = tf.layers.conv1d(space_to_weights, filters=n_features_input_all, kernel_size=1)
        space_to_weights_kernel = tf.reshape(space_to_weights_kernel, shape=(n_batch, n_max_entries, num_neighbors, -1))

        space_to_weights_bias = tf.layers.conv1d(space_to_weights, filters=1, kernel_size=1)
        space_to_weights_bias = tf.squeeze(tf.reshape(space_to_weights_bias, shape=(n_batch, n_max_entries, num_neighbors, -1)), axis=-1)

        output = tf.reduce_sum(tf.reduce_sum(tf.multiply(space_to_weights_kernel, gathered_all), axis=-1) + space_to_weights_bias, axis=-1)
        outputs.append(output)

    output = [tf.expand_dims(x, axis=-1) for x in outputs]
    output = tf.concat(output, axis=-1)
    output = tf.nn.relu(output)

    return construct_sparse_io_dict(output, spatial_features_global, spatial_features_local, num_entries)




