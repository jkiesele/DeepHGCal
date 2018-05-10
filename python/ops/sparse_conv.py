import tensorflow as tf


def sparse_conv_delta(A, B):
    """
    A-B

    :param A: A is of shape [B,E,N,F]
    :param B: B is of shape [B,E,F]
    :return:
    """

    return A - tf.expand_dims(B, axis=2)


def sparse_conv(space_features, all_features, neighbor_matrix, output_all=15):
    """
    Defines sparse convolution layer

    :param space_features: Space like features. Should be of shape [batch_size, num_entries, num_features]
    :param all_features: All features.  Should be of shape [batch_size, num_entries, num_features]
    :param neighbor_matrix: Matrix which contains nearest neighbors of each entry in a row. Should be of shape
                             [batch_size, num_entries, max_neighbors]

    :return: Computation graph which defines the sparse convolution
    """
    shape_space_features = space_features.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_neighbor_matrix = neighbor_matrix.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_max_neighbors = shape_neighbor_matrix[2]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]

    # All of these tensors should be 3-dimensional
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_neighbor_matrix) == 3

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0] == shape_neighbor_matrix[0]
    assert shape_space_features[1] == shape_all_features[1] == shape_neighbor_matrix[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert neighbor_matrix.dtype == tf.int32

    batch_range_vector = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1),axis=1), axis=1)
    batch_range_vector = tf.tile(batch_range_vector, [1,n_max_entries,n_max_neighbors,1])
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3)
    indexing_tensor = tf.concat([batch_range_vector, expanded_neighbor_matrix], axis=3)

    gathered_space_1 = tf.gather_nd(space_features, indexing_tensor) # [B,E,S] -> [B,E,5,S]
    # sparse_conv_delta(gathered_space_1, space_features) #  [B,E,5,S] -> [B,E,5,S]
    delta_gathered_space_2 = tf.reshape(sparse_conv_delta(gathered_space_1, space_features), [n_batch, n_max_entries, -1])
    delta_space_flattened = tf.layers.dense(inputs=delta_gathered_space_2, units=n_features_input_all, activation=tf.nn.relu)

    dotted_space = tf.multiply(all_features, delta_space_flattened)

    concatenated_all_features = tf.concat((dotted_space, space_features), axis=2)

    all_pen_output = tf.layers.dense(inputs=concatenated_all_features, units=output_all, activation=tf.nn.relu)

    all_output = tf.layers.dense(
        tf.reshape(tf.gather_nd(all_pen_output, indexing_tensor), [n_batch, n_max_entries, -1]), units=output_all,
        activation=tf.nn.relu)

    gathered_space = tf.gather_nd(space_features, indexing_tensor)
    gathered_space_2 = sparse_conv_delta(gathered_space, space_features)

    multiplication_factor_space = tf.layers.dense(
        tf.reshape(tf.gather_nd(all_pen_output, indexing_tensor), [n_batch, n_max_entries, -1]), units=n_features_input_space,
        activation=tf.nn.relu)

    add_to_space = tf.multiply(tf.layers.dense(tf.reshape(gathered_space_2, [n_batch, n_max_entries, -1]), units=n_features_input_space,
        activation=tf.nn.relu), multiplication_factor_space)
    space_output = add_to_space + space_features

    return space_output, all_output


# TODO: space_features is not used
def sparse_conv_bare(space_features, all_features, neighbor_matrix, output_all=15):
    """
    Define very simple sparse convolution layer which doesn't have complex space/other features dot products

    :param space_features: Space like features. Should be of shape [batch_size, num_entries, num_features]
    :param all_features: All features.  Should be of shape [batch_size, num_entries, num_features]
    :param neighbor_matrix: Matrix which contains nearest neighbors of each entry in a row. Should be of shape
                             [batch_size, num_entries, max_neighbors]

    :return: Computation graph which defines the sparse convolution
    """
    shape_space_features = space_features.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_neighbor_matrix = neighbor_matrix.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_max_neighbors = shape_neighbor_matrix[2]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]

    # All of these tensors should be 3-dimensional
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_neighbor_matrix) == 3

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0] == shape_neighbor_matrix[0]
    assert shape_space_features[1] == shape_all_features[1] == shape_neighbor_matrix[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert neighbor_matrix.dtype == tf.int32

    batch_range_vector = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1),axis=1), axis=1)
    batch_range_vector = tf.tile(batch_range_vector, [1,n_max_entries,n_max_neighbors,1])
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3)
    indexing_tensor = tf.concat([batch_range_vector, expanded_neighbor_matrix], axis=3)

    gathered_all = tf.reshape(tf.gather_nd(all_features, indexing_tensor), [n_batch, n_max_entries, -1])  # [B,E,F] -> [B,E,5*F]

    return tf.layers.dense(gathered_all, units=output_all, activation=tf.nn.relu)



def sparse_conv_2(space_features, all_features, neighbor_matrix, output_all=15):
    """
    Defines sparse convolution layer

    :param space_features: Space like features. Should be of shape [batch_size, num_entries, num_features]
    :param all_features: All features.  Should be of shape [batch_size, num_entries, num_features]
    :param neighbor_matrix: Matrix which contains nearest neighbors of each entry in a row. Should be of shape
                             [batch_size, num_entries, max_neighbors]

    :return: Computation graph which defines the sparse convolution
    """
    shape_space_features = space_features.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()
    shape_neighbor_matrix = neighbor_matrix.get_shape().as_list()

    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]
    n_max_neighbors = shape_neighbor_matrix[2]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_space_features[2]

    # All of these tensors should be 3-dimensional
    assert len(shape_space_features) == 3 and len(shape_all_features) == 3 and len(shape_neighbor_matrix) == 3

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert shape_space_features[0] == shape_all_features[0] == shape_neighbor_matrix[0]
    assert shape_space_features[1] == shape_all_features[1] == shape_neighbor_matrix[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert neighbor_matrix.dtype == tf.int32

    batch_range_vector = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1),axis=1), axis=1)
    batch_range_vector = tf.tile(batch_range_vector, [1,n_max_entries,n_max_neighbors,1])
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3)
    indexing_tensor = tf.concat([batch_range_vector, expanded_neighbor_matrix], axis=3)

    gathered_space_1 = tf.gather_nd(space_features, indexing_tensor) # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, space_features) # [B,E,5,S]

    weighting_factor_for_all_features = tf.reshape(delta_space, [n_batch, n_max_entries, -1])
    weighting_factor_for_all_features = tf.layers.dense(inputs=weighting_factor_for_all_features, units=n_max_neighbors, activation=tf.nn.relu) # [B,E,N]
    weighting_factor_for_all_features = tf.expand_dims(weighting_factor_for_all_features, axis=3)  # [B,E,N] - N = neighbors

    gathered_all = tf.gather_nd(all_features, indexing_tensor)  # [B,E,5,F]

    gathered_all_dotted = tf.concat((gathered_all * weighting_factor_for_all_features, gathered_all), axis=3)  # [B,E,5,2*F]
    pre_output = tf.layers.dense(gathered_all_dotted, output_all, activation=tf.nn.relu)
    output = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), output_all, activation=tf.nn.relu)

    weighting_factor_for_spatial_features = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), n_max_neighbors, activation=tf.nn.relu)
    weighting_factor_for_spatial_features = tf.expand_dims(weighting_factor_for_spatial_features, axis=3)

    spatial_output = space_features + tf.reduce_mean(delta_space * weighting_factor_for_spatial_features, axis=2)

    return output, spatial_output
