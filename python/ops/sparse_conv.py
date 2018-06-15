import tensorflow as tf
from .neighbors import indexing_tensor


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
        'num_entries' : num_entries
    }


@tf.custom_gradient
def gradient_scale_down(x):
  def grad(dy):
    return dy * 0.01
  return tf.identity(x), grad


@tf.custom_gradient
def gradient_scale_up(x):
  def grad(dy):
    return dy * 100
  return tf.identity(x), grad


def sparse_conv(sparse_dict, num_neighbors=10, output_all=15):
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

    gathered_space_1 = tf.gather_nd(spatial_features_global, _indexing_tensor)  # [B,E,5,S]
    delta_space = sparse_conv_delta(gathered_space_1, spatial_features_global)  # [B,E,5,S]

    spatial_features_local_gathered = tf.gather_nd(spatial_features_local, _indexing_tensor)

    weighting_factor_for_all_features = tf.reshape(delta_space, [n_batch, n_max_entries, -1])
    weighting_factor_for_all_features = tf.concat(
        (weighting_factor_for_all_features, tf.reshape(spatial_features_local_gathered, [n_batch, n_max_entries, -1])), axis=2)
    weighting_factor_for_all_features = gradient_scale_down(weighting_factor_for_all_features)

    weighting_factor_for_all_features = tf.layers.dense(inputs=weighting_factor_for_all_features, units=n_max_neighbors,
                                                        activation=tf.nn.softmax)  # [B,E,N]

    weighting_factor_for_all_features = gradient_scale_up(weighting_factor_for_all_features)

    weighting_factor_for_all_features = tf.clip_by_value(weighting_factor_for_all_features, 0, 1e5)
    weighting_factor_for_all_features = tf.expand_dims(weighting_factor_for_all_features,
                                                           axis=3)  # [B,E,N] - N = neighbors

    gathered_all = tf.gather_nd(all_features, _indexing_tensor)  # [B,E,5,F]

    gathered_all_dotted = gathered_all * weighting_factor_for_all_features# [B,E,5,2*F]
    # pre_output = tf.layers.dense(gathered_all, output_all, activation=tf.nn.relu)

    output = tf.layers.dense(tf.reshape(gathered_all_dotted, [n_batch, n_max_entries, -1]), output_all, activation=tf.nn.relu, )

    weighting_factor_for_spatial_features = tf.reshape(gathered_all_dotted, [n_batch, n_max_entries, -1])
    weighting_factor_for_spatial_features = gradient_scale_down(weighting_factor_for_spatial_features)

    weighting_factor_for_spatial_features = tf.layers.dense(weighting_factor_for_spatial_features,
                                                            n_max_neighbors,
                                                            activation=tf.nn.softmax)
    weighting_factor_for_spatial_features = gradient_scale_up(weighting_factor_for_spatial_features)

    weighting_factor_for_spatial_features = tf.clip_by_value(weighting_factor_for_spatial_features, 0, 1e5)
    weighting_factor_for_spatial_features = tf.expand_dims(weighting_factor_for_spatial_features, axis=3)

    spatial_output = spatial_features_global + tf.reduce_mean(delta_space * weighting_factor_for_spatial_features, axis=2)
    spatial_output_local = spatial_features_local + tf.reduce_mean(tf.gather_nd(spatial_features_local, _indexing_tensor) * weighting_factor_for_spatial_features, axis=2)

    # TODO: Confirm if this is done correctly
    mask = tf.cast(tf.expand_dims(tf.sequence_mask(num_entries, maxlen=n_max_entries), axis=2), tf.float32)
    output = output * mask
    spatial_output = spatial_output * mask
    spatial_output_local = spatial_output_local * mask

    return construct_sparse_io_dict(output, spatial_output, spatial_output_local, num_entries)


def sparse_merge_flat(sparse_dict, combine_three=True):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    shape_space_features = spatial_features_global.get_shape().as_list()
    n_batch = shape_space_features[0]
    n_max_entries = shape_space_features[1]

    mask = tf.cast(tf.expand_dims(tf.sequence_mask(num_entries, maxlen=n_max_entries), axis=2), tf.float32)
    nonzeros = tf.count_nonzero(mask, axis=1, dtype=tf.float32)

    flattened_features_all = tf.reshape(all_features, [n_batch, -1])
    flattened_features_spatial_features_global = tf.reshape(spatial_features_global, [n_batch, -1])
    flattened_features_spatial_features_local = tf.reshape(spatial_features_local, [n_batch, -1])

    if combine_three:
        output = tf.concat([flattened_features_all, flattened_features_spatial_features_global, flattened_features_spatial_features_local], axis=-1)
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
    batch_range = tf.tile(batch_range, [1,num_entries_result, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)

    out_all_features = tf.gather_nd(all_features, _indexing_tensor)
    out_spatial_features_global = tf.gather_nd(spatial_features_global, _indexing_tensor)
    out_spatial_features_local = tf.gather_nd(spatial_features_local, _indexing_tensor)

    num_entries = tf.minimum(tf.ones(shape=[n_batch], dtype=tf.int64) * num_entries_result, num_entries)

    return construct_sparse_io_dict(out_all_features, out_spatial_features_global, out_spatial_features_local, num_entries)


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
    batch_range = tf.tile(batch_range, [1,result_max_entires, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)

    out_all_features = tf.gather_nd(all_features, _indexing_tensor)
    out_spatial_features_global = tf.gather_nd(spatial_features_global, _indexing_tensor)
    out_spatial_features_local = tf.gather_nd(spatial_features_local, _indexing_tensor)

    num_entries_reduced = tf.cast(num_entries / factor, tf.int64)

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