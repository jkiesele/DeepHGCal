import tensorflow as tf


# just for testing

def printLayerStuff(l, desc_str):
    l=tf.Print(l,[l], desc_str+" ",summarize=2000)
    return l

def sparse_conv_delta(A, B):
    """
    A-B

    :param A: A is of shape [B,E,N,F]
    :param B: B is of shape [B,E,F]
    :return:
    """

    return A - tf.expand_dims(B, axis=2)


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
    assert neighbor_matrix.dtype == tf.int64

    batch_range_vector = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.range(0, n_batch, dtype=tf.int64), axis=1),axis=1), axis=1)
    batch_range_vector = tf.tile(batch_range_vector, [1,n_max_entries,n_max_neighbors,1])
    expanded_neighbor_matrix = tf.expand_dims(neighbor_matrix, axis=3)
    indexing_tensor = tf.concat([batch_range_vector, expanded_neighbor_matrix], axis=3)

    gathered_all = tf.reshape(tf.gather_nd(all_features, indexing_tensor), [n_batch, n_max_entries, -1])  # [B,E,F] -> [B,E,5*F]

    return tf.layers.dense(gathered_all, units=output_all, activation=tf.nn.relu), space_features


def sparse_conv_2(space_features, all_features, indexing_tensor, output_all=15, weight_init_width=1e-4,
                  merge_space_and_colour=False):
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
    shape_indexing_tensor = indexing_tensor.get_shape().as_list()

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
    assert indexing_tensor.dtype == tf.int64


    gathered_space_1 = tf.gather_nd(space_features, indexing_tensor) # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, space_features) # [B,E,5,S]
    

    weighting_factor_for_all_features = tf.reshape(delta_space, [n_batch, n_max_entries, -1])
    # this is a small correction
    weighting_factor_for_all_features = tf.layers.dense(inputs=weighting_factor_for_all_features, units=n_max_neighbors, 
                                                        activation=tf.nn.leaky_relu,
                                                        kernel_initializer=tf.random_normal_initializer(mean=weight_init_width, stddev=weight_init_width),
                                                        bias_initializer=tf.random_normal_initializer(mean=weight_init_width, stddev=weight_init_width)) # [B,E,N]
    
    weighting_factor_for_all_features = tf.clip_by_value(weighting_factor_for_all_features, 0, 1e5)
    weighting_factor_for_all_features = 1 + tf.expand_dims(weighting_factor_for_all_features, axis=3)  # [B,E,N] - N = neighbors

    
    gathered_all = tf.gather_nd(all_features, indexing_tensor)  # [B,E,5,F]

    gathered_all_dotted = tf.concat((gathered_all * weighting_factor_for_all_features, gathered_all), axis=3)  # [B,E,5,2*F]
    pre_output = tf.layers.dense(gathered_all_dotted, output_all, activation=tf.nn.relu)
    output = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), output_all, activation=tf.nn.relu,)

    weighting_factor_for_spatial_features = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), n_max_neighbors, 
                                                            activation=tf.nn.leaky_relu, 
                                                        kernel_initializer=tf.random_normal_initializer(mean=weight_init_width, stddev=weight_init_width),
                                                        bias_initializer=tf.random_normal_initializer(mean=weight_init_width, stddev=weight_init_width))
    
    weighting_factor_for_spatial_features = tf.clip_by_value(weighting_factor_for_spatial_features, 0, 1e5)
    weighting_factor_for_spatial_features = 1 + tf.expand_dims(weighting_factor_for_spatial_features, axis=3)

    spatial_output = space_features + tf.reduce_mean(delta_space * weighting_factor_for_spatial_features, axis=2)


    if merge_space_and_colour:
        output=tf.concat([output, spatial_output], axis=-1)
        return tf.layers.dense(output, output_all, activation=tf.nn.relu)
    else:
        return output, spatial_output


def sparse_max_pool(pooling_features, num_entries_result, graphs):
    shape_spatial_features = pooling_features.get_shape().as_list()

    n_batch = shape_spatial_features[0]

    # All of these tensors should be 3-dimensional
    assert len(shape_spatial_features) == 3

    # Neighbor matrix should be int as it should be used for indexing
    assert pooling_features.dtype == tf.float64 or pooling_features.dtype == tf.float32

    _, I = tf.nn.top_k(tf.reduce_sum(tf.abs(pooling_features), axis=2), num_entries_result)
    I = tf.expand_dims(I, axis=2)

    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1,num_entries_result, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)

    return [tf.gather_nd(i, _indexing_tensor) for i in graphs]


