from ops.sparse_conv import *


def sparse_conv_make_neighbors_old(sparse_dict, num_neighbors=10, output_all=15, spatial_degree_non_linearity=1, n_transformed_spatial_features=10, propagrate_ahead=False):
    """
    Defines sparse convolutional layer
    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :return: Dictionary containing output which can be made input to the next layer
    """

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

    assert spatial_degree_non_linearity >= 1

    transformed_space_features = tf.concat([spatial_features_global], axis=2)

    for i in range(spatial_degree_non_linearity - 1):
        transformed_space_features = tf.layers.dense(transformed_space_features, n_transformed_spatial_features, activation=tf.nn.relu)

    transformed_space_features = tf.layers.dense(transformed_space_features, n_transformed_spatial_features, activation=None, kernel_initializer=NoisyEyeInitializer)
    # transformed_space_features = tf.layers.dense(transformed_space_features, 10, activation=tf.nn.relu)

    _indexing_tensor, distance = indexing_tensor_2(transformed_space_features, num_neighbors)

    gathered_all = tf.gather_nd(all_features, _indexing_tensor) * tf.expand_dims(tf.nn.softmax(-distance), axis=3) # [B,E,5,F]

    pre_output = tf.layers.dense(gathered_all, output_all, activation=tf.nn.relu)
    output = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), output_all, activation=tf.nn.relu)

    return construct_sparse_io_dict(output, transformed_space_features if propagrate_ahead else spatial_features_global, spatial_features_local, num_entries)
