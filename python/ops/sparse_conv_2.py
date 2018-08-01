import tensorflow as tf
from .neighbors import indexing_tensor
from .neighbors import sort_last_dim_tensor


# just for testing

def gauss_activation(x, name=None):
    return tf.exp(-x * x / 4, name)


def printLayerStuff(l, desc_str):
    l = tf.Print(l, [l], desc_str + " ", summarize=2000)
    return l


def sparse_conv_delta(A, B):
    """
    A-B

    :param A: A is of shape [B,E,N,F]
    :param B: B is of shape [B,E,F]
    :return:
    """

    return A - tf.expand_dims(B, axis=2)


def construct_sparse_io_dict(all_features, spatial_features_global, spatial_features_local, num_entries):
    """
    Constructs dictionary for io of sparse convolution layers

    :param all_features: All features tensor.  Should be of shape [batch_size, num_entries, num_features]
    :param spatial_features_global: Space like features tensor. Should be of shape [batch_size, num_entries, num_features]
    :param spatial_features_local: Space like features tensor (sensor sizes etc). Should be of shape [batch_size, num_entries, num_features]
    :param num_entries: Number of entries tensor for each batch entry.
    :return: dictionary in the format of the sparse conv layer
    """
    return {
        'all_features': all_features,
        'spatial_features_global': spatial_features_global,
        'spatial_features_local': spatial_features_local,
        'num_entries': num_entries
    }


@tf.custom_gradient
def gradient_scale_down(x):
    def grad(dy):
        return dy * .01

    return tf.identity(x), grad


@tf.custom_gradient
def gradient_scale_up(x):
    def grad(dy):
        return dy * 100

    return tf.identity(x), grad


def sparse_conv(sparse_dict, num_neighbors=10, num_filters=15, n_prespace_conditions=3):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param num_filters: Number of output features for color like outputs
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

    print("Indexing tensor shape", _indexing_tensor.shape)
    gathered_spatial = tf.gather_nd(spatial_features_global, _indexing_tensor)  # [B,E,5,S]

    print("Gathered spatial shape", spatial_features_global.shape, gathered_spatial.shape)
    delta_space = sparse_conv_delta(gathered_spatial, spatial_features_global)  # [B,E,5,S]

    spatial_features_local_gathered = tf.gather_nd(spatial_features_local, _indexing_tensor)
    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,N,A]

    """
    Hint: (from next line onward)
        B = Batch Size
        E = Num max entries
        N = Number of neighbors
        S = number of spatial features (both local and global combined)
        F = number of filters
        M = Number of pre-space conditions
    """
    spatial_features_concatenated = tf.concat([delta_space, spatial_features_local_gathered], axis=-1)  # [B, E, N, S]

    # delta_space = tf.concat([delta_space,delta_space*delta_space], axis=-1)


    sorting_condition = tf.layers.dense(inputs=spatial_features_concatenated,
                                          units=num_filters,
                                          activation=tf.nn.relu)   # [B, E, N, F]

    space_condition = tf.layers.dense(inputs=spatial_features_concatenated,
                                          units=(num_filters*n_prespace_conditions),
                                          activation=tf.nn.relu)  # [B, E, N, F*M]
    space_condition = tf.reshape(space_condition, [n_batch, n_max_entries, num_neighbors, num_filters, -1])

    filter_outputs = []
    for i in range(num_filters):
        filter_input = space_condition[:,:,:, i, :]
        filter_output = tf.layers.dense(inputs=filter_input, units=1, activation=gauss_activation)
        tf.expand_dims(filter_output, dim=3)
        filter_outputs.append(filter_output)

    space_condition = tf.concat(filter_outputs, axis=3)
    sorting_values = tf.multiply(space_condition, sorting_condition) # [B, E, N, F]

    filter_outputs = []
    for i in range(num_filters):
        filter_neighbor_values = sorting_values[..., i] # [B, E, N]
        sorting_tensor = sort_last_dim_tensor(filter_neighbor_values)
        filter_input = tf.reshape(tf.gather_nd(gathered_all, sorting_tensor), shape=[n_batch, n_max_entries, -1])
        filter_output = tf.layers.dense(inputs=filter_input, units=1, activation=tf.nn.relu)
        filter_outputs.append(filter_output)

    color_like_output = tf.concat(filter_outputs, axis=-1)

    return construct_sparse_io_dict(color_like_output , spatial_features_global, spatial_features_local, num_entries)


def sparse_merge_flat(sparse_dict, combine_three=True):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    shape_space_features = spatial_features_global.get_shape().as_list()
    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]

    # order the output according to the spatial global distance to zero >= 0
    distancetozero = tf.expand_dims(tf.reduce_sum(spatial_features_global *
                                                  spatial_features_global, axis=2), axis=2)
    _, I = tf.nn.top_k(tf.reduce_sum(distancetozero, axis=2), n_max_entries)
    I = tf.expand_dims(I, axis=2)
    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1, n_max_entries, 1])
    ordering = tf.concat([batch_range, I], axis=2)

    gathered_feat = tf.gather_nd(all_features, ordering)
    gathered_spatial_global = tf.gather_nd(spatial_features_global, ordering)
    gathered_spatial_local = tf.gather_nd(spatial_features_local, ordering)

    # mask not used?!
    # mask = tf.cast(tf.expand_dims(tf.sequence_mask(num_entries, maxlen=n_max_entries), axis=2), tf.float32)
    # nonzeros = tf.count_nonzero(mask, axis=1, dtype=tf.float32)

    flattened_features_all = tf.reshape(gathered_feat, [n_batch, -1])
    flattened_features_spatial_features_global = tf.reshape(gathered_spatial_global, [n_batch, -1])
    flattened_features_spatial_features_local = tf.reshape(gathered_spatial_local, [n_batch, -1])

    if combine_three:
        output = tf.concat([flattened_features_all, flattened_features_spatial_features_global,
                            flattened_features_spatial_features_local], axis=-1)
    else:
        output = flattened_features_all

    return output


def sparse_max_pool(sparse_dict, num_entries_result):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    shape_spatial_features = spatial_features_global.get_shape().as_list()
    shape_spatial_features_local = spatial_features_local.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()

    n_batch = shape_spatial_features[0]
    n_max_entries = shape_spatial_features[1]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_spatial_features[2]

    # All of these tensors should be 3-dimensional
    assert len(shape_spatial_features) == 3

    # Neighbor matrix should be int as it should be used for indexing
    assert all_features.dtype == tf.float64 or all_features.dtype == tf.float32

    _, I = tf.nn.top_k(tf.reduce_sum(all_features, axis=2), num_entries_result)
    I = tf.expand_dims(I, axis=2)

    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1, num_entries_result, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)

    out_all_features = tf.gather_nd(all_features, _indexing_tensor)
    out_spatial_features_global = tf.gather_nd(spatial_features_global, _indexing_tensor)
    out_spatial_features_local = tf.gather_nd(spatial_features_local, _indexing_tensor)

    num_entries = tf.minimum(tf.ones(shape=[n_batch], dtype=tf.int64) * num_entries_result, num_entries)

    return construct_sparse_io_dict(out_all_features, out_spatial_features_global, out_spatial_features_local,
                                    num_entries)


def sparse_max_pool_factored(sparse_dict, factor):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    shape_spatial_features = spatial_features_global.get_shape().as_list()
    shape_spatial_features_local = spatial_features_local.get_shape().as_list()
    shape_all_features = all_features.get_shape().as_list()

    n_batch = shape_spatial_features[0]
    n_max_entries = shape_spatial_features[1]
    n_features_input_all = shape_all_features[2]
    n_features_input_space = shape_spatial_features[2]

    # All of these tensors should be 3-dimensional
    assert len(shape_spatial_features) == 3

    # Neighbor matrix should be int as it should be used for indexing
    assert all_features.dtype == tf.float64 or all_features.dtype == tf.float32

    result_max_entires = int(n_max_entries / factor)

    _, I = tf.nn.top_k(tf.reduce_sum(all_features, axis=2), result_max_entires)
    I = tf.expand_dims(I, axis=2)

    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1, result_max_entires, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)

    out_all_features = tf.gather_nd(all_features, _indexing_tensor)
    out_spatial_features_global = tf.gather_nd(spatial_features_global, _indexing_tensor)
    out_spatial_features_local = tf.gather_nd(spatial_features_local, _indexing_tensor)

    num_entries_reduced = tf.cast(num_entries / factor, tf.int64)
    num_entries_reduced = tf.where(tf.equal(num_entries_reduced, tf.zeros_like(num_entries_reduced)),
                                   tf.zeros_like(num_entries_reduced) + 1, num_entries_reduced)

    mask = tf.cast(tf.expand_dims(tf.sequence_mask(num_entries_reduced, maxlen=result_max_entires), axis=2), tf.float32)
    #
    # num_entries = tf.minimum(tf.ones(shape=[n_batch], dtype=tf.int64) * num_entries_result, num_entries)

    return construct_sparse_io_dict(mask * out_all_features, mask * out_spatial_features_global,
                                    mask * out_spatial_features_local, num_entries_reduced)


def sparse_conv_bare(sparse_dict, num_neighbors=10, output_all=15, weight_init_width=1e-4):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :param weight_init_width: TODO: Fill this
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

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    pre_output = tf.layers.dense(gathered_all, output_all, activation=tf.nn.relu)
    output = tf.layers.dense(tf.reshape(pre_output, [n_batch, n_max_entries, -1]), output_all, activation=tf.nn.relu)

    return construct_sparse_io_dict(output, spatial_features_global, spatial_features_local, num_entries)