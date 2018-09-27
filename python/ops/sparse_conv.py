import tensorflow as tf
from .neighbors import indexing_tensor, indexing_tensor_2, sort_last_dim_tensor
from ops.nn import *
import numpy as np
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import random_ops


def sprint(tensor,pstr):
    return tf.Print(tensor,[tensor],pstr)

#small width
def gauss_activation(x, name=None):
    return tf.exp(-x * x / 4, name)


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


@tf.custom_gradient
def gradient_off(x):

    def grad(dy):
        return dy * 0

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

    _, I = tf.nn.top_k(tf.reduce_max(all_features, axis=2), num_entries_result)
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

    _, I = tf.nn.top_k(tf.reduce_max(all_features, axis=2), result_max_entires)
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


def find_filter_weights(x, num_outputs=10, activation=tf.nn.relu):
    X = []
    for j in range(num_outputs):
        X.append(tf.expand_dims(filter_wise_dense(x), axis=-1))

    return tf.concat(X, axis=-1)


# input [n_batch, n_max_entries, num_neighbors, n_space_feat]
def apply_space_transformations(x, depth, num_filters, n_outputs, nodes_relu=5, nodes_gauss=7):
    n_batch = x.get_shape().as_list()[0]
    n_max_entries = x.get_shape().as_list()[1]
    num_neighbors =  x.get_shape().as_list()[2]
    
    weight_values = tf.layers.dense(inputs=x,
                                          units=num_filters*(nodes_relu+nodes_gauss),
                                          activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0.02, 0.02))   # [B, E, N, F*C]
    weight_values=tf.nn.leaky_relu(weight_values,alpha=0.01)
    
    weight_values = tf.reshape(weight_values, [n_batch, n_max_entries, num_neighbors, 
                                                       num_filters, (nodes_relu+nodes_gauss)])
    
    X = []
    for f in range(num_filters):
        this_input=weight_values[:,:,:,f,:]
        filter_network=[]
        for j in range(depth):
            x_local_relu = tf.layers.dense(this_input, units=nodes_relu,activation=None)
            x_local_relu=tf.nn.leaky_relu(x_local_relu,alpha=0.01)
            x_local_gauss = tf.layers.dense(this_input, units=nodes_gauss,activation=gauss_activation)
            x_local = tf.concat([x_local_relu,x_local_gauss], axis=-1)
            #print('x_local shape filter', f,'depth', j , x_local.shape)
            this_input=x_local
        filter_out = tf.layers.dense(inputs=this_input, units=n_outputs,activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0.02, 0.02))
        #filter_out=tf.nn.softsign(filter_out)#,alpha=0.01)
        #print('filter_out shape filter', f , filter_out.shape)
        X.append(filter_out)
    allout = tf.concat(X,axis=-1)
    #print('allout shape', allout.shape)
    allout = tf.reshape(allout, [n_batch, n_max_entries, num_neighbors, 
                                                       num_filters, n_outputs])
    
    #print('allout shape b', allout.shape)
    return allout
    
    

def sparse_conv_2(sparse_dict, num_neighbors=8, num_filters=16, n_prespace_conditions=4,
                  pre_space_relu=6, pre_space_gauss=3):
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
    n_features_input_space_local = shape_space_features_local[2]
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

    
    # So far everything is the same as before.
    # Now, the general idea is as follows:
    # A concolutional filter is something like f(sum (weights*colours)),
    # where f is the activation function, and the weights are learnt.
    # It is important that the weight always corresponds to the same relative input coordinate
    # (which is automatic in the convolutional grid sense).
    # 
    # Now, if we think about it, we already had everything at hand:
    # We learnt the sorting condition to sort the neighbours before the weights 
    # were supposed to be applied.
    # However, we can think of the sorting condition already as weights that are learnt.
    # Then, they are also automatically applied to the right neighbour.
    # The subsequent sum is anyway invariant w.r.t. the order of the neighbours, so it is
    # the same as a convolutional filter, with the expection that the weight learning/calculation
    # is a bit more evolved.
    #
    # And additional thing I added was to apply different weights per colour (as for a standard conv filter)
    # Therefore, there are all these '*n_features_input_all'.
    #
    # The whole thing is significantly faster than before.
    # I have added some comments where I think things should be improved
    #
    
    
    # [B, E, N, S]
    weight_values = spatial_features_concatenated
    
    # Given this corresponds to the weight learning, I think it makes sense to make this a
    # bit deeper. Using something like filter_wise_dense, but as
    # filter_and_colour_wise dense (TBI), also with an adjustable node length in the intermediate
    # layers. Note, that the last layer should not have relu as activation because it 
    # allows for positive weights, only. Maybe linear works, maybe some other stuff.
    # Here I put softsign just as a reminder.
    #
    # In a similar manner a branch of this calculation can include the gaussian activation in parallel
    # In the end, they are combined in filter and colour (C) wise output of 
    # [B, E, N, F, C], where F and C are dimentions with non shared weights
    #
    
    weight_values = apply_space_transformations(spatial_features_concatenated, n_prespace_conditions,
                                                 num_filters,n_features_input_all,nodes_relu=pre_space_relu,nodes_gauss=pre_space_gauss)

    weight_values = tf.transpose(weight_values, perm=[0, 1, 3, 2, 4]) # [B, E, F, N, C]
    
    print('weight_values shape ', weight_values.shape)
    
    inputs=tf.expand_dims(gathered_all, axis=2)
    
    print('inputs shape ',inputs.shape)
    color_like_output = tf.multiply(inputs, weight_values)
    print('color_like_output shape ',color_like_output.shape)
    #sum: [B, E, F, N, C] -> [B, E, F]
    # colour_reduced could be interesting input for the space transformation!
    colour_reduced = tf.reduce_sum(color_like_output, axis=-1)
    color_like_output = tf.reduce_sum(colour_reduced, axis=-1)
    color_like_output = tf.nn.leaky_relu(color_like_output,alpha=0.01)

    print('color_like_output.shape b ', color_like_output.shape)
    
    return construct_sparse_io_dict(color_like_output , spatial_features_global, spatial_features_local, num_entries)


class NoisyEyeInitializer(Initializer):
  """Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    dtype: The data type. Only floating point types are supported.
  """

  def __init__(self, low=0.0, up=1.0, seed=None, dtype=tf.float32):
    self.low = low
    self.high = up
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    assert len(shape) == 2
    return tf.eye(shape[0], shape[1]) + random_ops.random_uniform(
        shape, self.low, self.high, dtype, seed=self.seed)

  def get_config(self):
    return {
        "low": self.low,
        "high": self.high,
        "seed": self.seed,
        "dtype": self.dtype.name
    }

def noisy_eye_initializer():
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 2:
            eye = np.eye(shape[0], shape[1])+np.random.uniform(0,0.1, size=(shape[0], shape[1]))
            eye = eye.astype(dtype=dtype)
            return tf.constant_op.constant(eye)
        else:
            raise ValueError('Invalid shape')
    return _initializer


def sparse_conv_make_neighbors(sparse_dict, num_neighbors=10, output_all=15, spatial_degree_non_linearity=1, n_transformed_spatial_features=10, propagrate_ahead=False):
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




# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...
# loop implementations down here...

def check_inputs(colours_in, space_global, space_local, num_entries, indexing_tensor):
    
    assert len(space_global.get_shape().as_list()) == 3 and len(colours_in.get_shape().as_list()) == 3 and len(indexing_tensor.get_shape().as_list()) == 4

    # First dimension is batch, second is number of entries, hence these two should be same for all
    assert space_global.get_shape().as_list()[0] == space_global.get_shape().as_list()[0]
    assert space_global.get_shape().as_list()[1] == space_local.get_shape().as_list()[1]

    # Neighbor matrix should be int as it should be used for indexing
    assert indexing_tensor.dtype == tf.int64



def apply_space_transformations_loop(x, depth, num_filters, n_outputs, nodes_relu=5, nodes_gauss=7):
    n_batch = x.get_shape().as_list()[0]
    n_max_entries = x.get_shape().as_list()[1]
    
    # x is: B x E x S
    
    weight_values = tf.layers.dense(inputs=x,
                                          units=num_filters*(nodes_relu+nodes_gauss),
                                          activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0.02, 0.02))   # [B, E, N, F*C]
    weight_values=tf.nn.leaky_relu(weight_values,alpha=0.01)
    
    weight_values = tf.reshape(weight_values, [n_batch, n_max_entries, 
                                                       num_filters, (nodes_relu+nodes_gauss)])
    
    X = []
    for f in range(num_filters):
        this_input=weight_values[:,:,f,:]
        filter_network=[]
        for j in range(depth):
            x_local_relu = tf.layers.dense(this_input, units=nodes_relu,activation=None)
            x_local_relu=tf.nn.leaky_relu(x_local_relu,alpha=0.01)
            x_local_gauss = tf.layers.dense(this_input, units=nodes_gauss,activation=gauss_activation)
            x_local = tf.concat([x_local_relu,x_local_gauss], axis=-1)
            #print('x_local shape filter', f,'depth', j , x_local.shape)
            this_input=x_local
        filter_out = tf.layers.dense(inputs=this_input, units=n_outputs,activation=None,
                                     kernel_initializer=tf.random_normal_initializer(0.02, 0.02))
        #filter_out=tf.nn.softsign(filter_out)#,alpha=0.01)
        #print('filter_out shape filter', f , filter_out.shape)
        X.append(filter_out)
    allout = tf.concat(X,axis=-1)
    #print('allout shape', allout.shape)
    allout = tf.reshape(allout, [n_batch, n_max_entries,num_filters, n_outputs])
    
    #print('allout shape b', allout.shape)
    return allout


def neighbour_loop(colours_in,space_global,space_local,neighbour_indexing,iterator_index,filters_output, depth, nodes_relu=5, nodes_gauss=7):
    
    # selected_index: neighbour that is being processed
    # neighbour_indexing: indexing tensor
    # iterator_index iterator index of indexing tensor


    n_batch = space_global.get_shape().as_list()[0]
    n_max_entries = space_global.get_shape().as_list()[1]
    
    n_colours_in = colours_in.get_shape().as_list()[2]
    n_space_global = space_global.get_shape().as_list()[2]
    n_space_local = space_local.get_shape().as_list()[2]
    n_neighbours = neighbour_indexing.get_shape().as_list()[2]
    
    num_filters=filters_output.get_shape().as_list()[2]
    #tested
    its=neighbour_indexing[:,:,iterator_index+1]
    
    
    #change this to include more space features, just for testing
    #FIXME
    selected_neighbour_space=tf.gather_nd(space_local, its)
    
    neighbour_weight=apply_space_transformations_loop(selected_neighbour_space, depth, num_filters, n_colours_in,nodes_relu,nodes_gauss)
    print('neighbour_weight.shape',neighbour_weight.shape)
    
    # filter: B X E X F , neighbour_weight: B X E X F X C , colours_in: B X E X C
    colours_in = tf.expand_dims(colours_in, axis=2)
    to_be_added = neighbour_weight*colours_in
    print('to_be_added.shape',to_be_added.shape)
    
    to_be_added = tf.reduce_sum(to_be_added,axis=-1)
    print('to_be_added.shape b',to_be_added.shape)
    print('filters_output.shape',filters_output.shape)
    filters_output += to_be_added
    # output = weight*colour
    # add to filter output
    iterator_index=tf.add(iterator_index,1)
    
    #iterator_index=sprint(iterator_index,'it')
    
    return [iterator_index,filters_output]




def sparse_conv_loop(sparse_dict, num_neighbors=8, num_filters=16, space_depth=4,
                  space_relu=6, space_gauss=3):
    """
    Defines sparse convolutional layer

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param num_filters: Number of output features for color like outputs
    :return: Dictionary containing output which can be made input to the next layer
    """

    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']

    neighbour_indexing = indexing_tensor(space_global, num_neighbors)

    check_inputs(colours_in, space_global, space_local, num_entries, neighbour_indexing)

    n_batch = space_global.get_shape().as_list()[0]
    n_max_entries = space_global.get_shape().as_list()[1]
    
    n_colours_in = colours_in.get_shape().as_list()[2]
    n_space_global = space_global.get_shape().as_list()[2]
    n_space_local = space_local.get_shape().as_list()[2]
    n_neighbours = neighbour_indexing.get_shape().as_list()[2]


    #create calculation with placeholders and then pass it to the neighbour loop?
    


    #apply to each of the neighbours same thing
    # colours_in,space_global,space_local,neighbour_indexing,iterator_index
    filter_output = tf.zeros([n_batch,n_max_entries,num_filters])
    iterator_index = tf.constant(0)
    c = lambda i,fo: tf.less(i, n_neighbours-1)
    b = lambda i,fo: neighbour_loop(colours_in,space_global,space_local,neighbour_indexing,i,fo, space_depth, space_relu, space_gauss)
    iterator_index,filter_output=tf.while_loop(c, b, [iterator_index,filter_output],parallel_iterations=1,swap_memory=True)


    
    return construct_sparse_io_dict(filter_output , space_global, space_local, num_entries)










