import tensorflow as tf
from .neighbors import indexing_tensor, indexing_tensor_2, sort_last_dim_tensor, get_sorted_vertices_ids
from ops.nn import *
import numpy as np
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import random_ops
import math 

def sprint(tensor,pstr):
    return tf.Print(tensor,[tensor],pstr)

#small width
def gauss_activation(x, name=None):
    return tf.exp(-x * x / 4, name)


def make_sequence(nfilters):
    isseq=(not hasattr(nfilters, "strip") and
            hasattr(nfilters, "__getitem__") or
            hasattr(nfilters, "__iter__"))
    
    if not isseq:
        nfilters=[nfilters]
    return nfilters

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
    Constructs dictionary for readers of sparse convolution layers

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

  def __init__(self, low=-0.01, up=0.01, seed=None, dtype=tf.float32):
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


def sparse_conv_mix_colours_to_space(sparse_dict):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']
                                                                                 
    spatial_features_global = tf.concat([spatial_features_global,all_features],axis=-1)
    return construct_sparse_io_dict(all_features, spatial_features_global, spatial_features_local, num_entries)

def sparse_conv_collapse(sparse_dict):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']
    return tf.concat([spatial_features_global,all_features , spatial_features_local],axis=-1)                                                                             
    
def sparse_conv_split_batch(sparse_dict,split):
    
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']
    
    first  = construct_sparse_io_dict(all_features[0:split,:,:], spatial_features_global[0:split,:,:], spatial_features_local[0:split,:,:], num_entries)
    second = construct_sparse_io_dict(all_features[split:,:,:], spatial_features_global[split:,:,:], spatial_features_local[split:,:,:], num_entries)
    return first,second

def sparse_conv_make_neighbors2(sparse_dict, num_neighbors=10, 
                               output_all=15, space_transformations=[10],
                               propagrate_ahead=False,
                               strict_global_space=True,name = None):
        
        
        
    #   sparse_dict, num_neighbors=10, 
    #                          output_all=15, spatial_degree_non_linearity=1, 
    #                          n_transformed_spatial_features=10, 
    #                          propagrate_ahead=False):
    """
    Defines sparse convolutional layer
    
    --> revise the space distance stuff

    :param sparse_dict: Dictionary containing input
    :param num_neighbors: An integer containing number of neighbors to pick + 1 (+1 is for yourself)
    :param output_all: Number of output features for color like outputs
    :return: Dictionary containing output which can be made input to the next layer
    """
    if name is None:
        name="sparse_conv_make_neighbors2"

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


    #add local space features here?
    transformed_space_features = tf.concat([spatial_features_global,spatial_features_local], axis=-1)
    #these are the same in each sample
    if strict_global_space:
        transformed_space_features = transformed_space_features[0,:,:]
        transformed_space_features = tf.expand_dims(transformed_space_features,axis=0)

    space_transformations = make_sequence(space_transformations)

    for i in range(len(space_transformations)):
        transformed_space_features = tf.layers.dense(transformed_space_features, space_transformations[i], 
                                                     activation=tf.nn.tanh, kernel_initializer=NoisyEyeInitializer,
                                                     name=name+"_sp_"+str(i))
            
    create_indexing_batch = n_batch
    if not strict_global_space:
        create_indexing_batch=-1

    breakdown_x = (tf.layers.dense(transformed_space_features, 1, activation=tf.nn.tanh) + 3.) * 2
    breakdown_y = tf.layers.dense(transformed_space_features, 1, activation=tf.nn.tanh)
    sorting = breakdown_x + breakdown_y

    _indexing_tensor, distance = indexing_tensor_2(sorting, num_neighbors,create_indexing_batch)
    
    #for future use
    if strict_global_space:
        transformed_space_features = tf.tile(transformed_space_features,[n_batch,1,1])
    
    #distance is strict positive
    inverse_distance =  1-tf.nn.softsign(-distance) # *float(num_neighbors)
    expanded_distance = tf.expand_dims(inverse_distance, axis=3)
   
   
    last_iteration = all_features
    output_all = make_sequence(output_all)
    for i in range(len(output_all)):
        f = output_all[i]
        gathered_all = tf.gather_nd(last_iteration, _indexing_tensor) * expanded_distance
        flattened_gathered = tf.reshape(gathered_all, [n_batch, n_max_entries, -1])
        last_iteration = tf.layers.dense(flattened_gathered, f, activation=tf.nn.relu, name = name+ str(f) + str(i)) 

    output_global_space = spatial_features_global
    if propagrate_ahead:
        output_global_space = tf.concat([transformed_space_features,spatial_features_global],axis=-1)
       
    return construct_sparse_io_dict(last_iteration, output_global_space, spatial_features_local, num_entries)

   
def sparse_conv_make_neighbors(sparse_dict, num_neighbors=10, 
                               output_all=15, spatial_degree_non_linearity=1, 
                               n_transformed_spatial_features=10, 
                               propagrate_ahead=False, name="sparse_conv_make_neighbors"):
    
    return sparse_conv_make_neighbors2(sparse_dict=sparse_dict, num_neighbors=num_neighbors, 
                               output_all=output_all, space_transformations=
                               [n_transformed_spatial_features for i in range(spatial_degree_non_linearity)],
                               propagrate_ahead=propagrate_ahead, name=name)

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


from ops.neighbors import euclidean_squared,n_range_tensor
def sparse_conv_full_adjecency(sparse_dict, nfilters, AdMat=None, iterations=1,spacetransform=-1,noutputfilters=-1):
    
    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']
    
    layerin = tf.concat([colours_in,space_local,space_global], axis=-1)      
    layerout=layerin                                                                     
    if  AdMat is  None:
        firstelement_space = space_global[0,:,:]
        firstelement_space = tf.expand_dims(firstelement_space,axis=0)
        if spacetransform>0:
            firstelement_space = tf.concat([firstelement_space,tf.expand_dims(space_local[0,:,:],axis=0)],axis=-1)
            firstelement_space = tf.layers.dense(firstelement_space, spacetransform,activation=tf.nn.relu,
                                                 kernel_initializer=NoisyEyeInitializer)
            firstelement_space = tf.layers.dense(firstelement_space,3,activation=None,
                                                 kernel_initializer=NoisyEyeInitializer)
        print('firstelement_space',firstelement_space.shape)
        AdMat = euclidean_squared(firstelement_space,firstelement_space)
        print('AdMat',AdMat.shape)
        #BxExE
        #normalise
        #AdMat=tf.Print(AdMat,[AdMat],'AdMat')
        maxAdMat = tf.reduce_max(tf.reduce_max(AdMat, axis=-1,keepdims=True),axis=-1,keepdims=True)
        AdMat = AdMat/maxAdMat
        #AdMat=tf.Print(AdMat,[AdMat],'AdMata')
        AdMat = (tf.zeros_like(AdMat)+1) - AdMat
        AdMat = tf.expand_dims(AdMat, axis=3)
        scaling = tf.reduce_sum(tf.reduce_mean(AdMat, axis=-1, keep_dims=False))/float(int(AdMat.get_shape()[1]))
        AdMat = AdMat / scaling 
        print('AdMat shape',AdMat.shape)
        #AdMat=tf.Print(AdMat,[AdMat],'AdMat2')
        #features = tf.concat([colours_in,space_local], axis=-1)
        
    
    print('features',layerout.shape)

    
    isseq=(not hasattr(nfilters, "strip") and
            hasattr(nfilters, "__getitem__") or
            hasattr(nfilters, "__iter__"))
    
    if not isseq:
        nfilters=[nfilters]
    
    for i in range(len(nfilters)):
        layerout = tf.layers.dense(layerout, nfilters[i], activation=tf.nn.relu)
    
    for i in range(iterations):
        layerout = tf.reshape(layerout,shape=[layerout.shape[0],AdMat.shape[1],1,nfilters[-1]])
        layerout = AdMat * layerout #tf.tile(transformed_features,[1,1,AdMat.shape[-2],1]) *1./2000.
        print('layerout ',layerout.shape)
        layerout = tf.reduce_sum(layerout,axis=-2)
    
    layerout = tf.concat([layerout,layerin],axis=-1)
    #new
    if noutputfilters<=0:
        noutputfilters=nfilters[-1]
    layerout = tf.layers.dense(layerout,noutputfilters,activation=tf.nn.relu)
    #layerout = tf.Print(layerout,[layerout],'layerout')
    
    print('layerout2 ',layerout.shape)
    
    return construct_sparse_io_dict(layerout , space_global, space_local, num_entries), AdMat

def make_batch_selection(ids):
    n_batch=ids.shape[0]
    n_vertices=ids.shape[1]
    ids = tf.cast(ids, dtype=tf.int64)
    batch=tf.range(n_batch, dtype=tf.int64)
    batch = tf.tile(batch[..., tf.newaxis, tf.newaxis], [1,n_vertices,1])
    select = tf.concat((batch, ids[..., tf.newaxis]), axis=-1)
    return select
    
    
    

#just for compat
def make_seed_selector(seed_ids):
    return make_batch_selection(seed_ids)

def normalise_distance_matrix(AdMat):
    #return 1-tf.nn.softsign(4*AdMat)
    #mat=tf.exp(-1*(tf.abs(AdMat)))
    mat=tf.nn.softmax(-tf.abs(AdMat))
    #mat=sprint(mat, "pstr")
    return mat


def sparse_conv_make_seeds(sparse_dict,
                           space_dimensions,
                           n_seeds = 3,
                           conv_kernels=[],
                           conv_filters=[]):
    '''
    seed finding based on conv layers applied to a (learnable) distance matrix in 2 dimensions
    '''
    
    assert len(conv_kernels)==len(conv_filters)
    
    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                         sparse_dict['spatial_features_global'], \
                                                         sparse_dict['spatial_features_local'], \
                                                         sparse_dict['num_entries']
    
    n_vertices = colours_in.shape[1]
    next_square = (int(math.sqrt(float(int(n_vertices+1)))+1))**2
    xy_dim = int(math.sqrt(next_square))
    print(next_square)
    n_batch = colours_in.shape[0]
    n_features = colours_in.shape[2]
    
    use_features = tf.concat([space_global,colours_in,space_local], axis=-1)
    dimension_breakdown= tf.layers.dense(use_features, space_dimensions,activation=tf.nn.tanh,
                                         kernel_initializer=NoisyEyeInitializer)
    # break it down to 2 dimensions to get a 2D conv grid
    breakdown_x = (tf.layers.dense(dimension_breakdown, 1,activation=tf.nn.tanh)+3.)*2
    breakdown_y = tf.layers.dense(dimension_breakdown, 1,activation=tf.nn.tanh)
    sorting = breakdown_x+breakdown_y
    sorting = tf.squeeze(sorting) #only one dimension
    ids = get_sorted_vertices_ids(sorting)
    all_features = tf.concat([dimension_breakdown,(breakdown_x-3.)/2.,breakdown_y], axis=-1)

    sorted_all_features = tf.gather_nd(all_features,ids)
    #sorted_all_features= tf.squeeze(sorted_all_features)
    print('sorted_all_features',sorted_all_features.shape)
    
    padded = tf.pad(sorted_all_features,[[0,0],[0,next_square-n_vertices],[0,0]],"CONSTANT")
    x =  tf.reshape(padded,[n_batch,xy_dim,xy_dim,-1] )
    print('x',x.shape)
    
    for i in range(len(conv_kernels)):
        x = tf.layers.conv2d(x,conv_filters[i],kernel_size=conv_kernels[i],padding='same',
                             activation=tf.nn.relu)
    
    x = tf.reshape(x, [n_batch,x.shape[1], -1])
    x = tf.reduce_sum(x,axis=-1)
    v, outidx = tf.nn.top_k(x,k=n_seeds)
    v = tf.nn.tanh(v)
    v = tf.expand_dims(v, axis=2)
    #translate back
    allids = tf.range(n_vertices)
    allids = tf.expand_dims(allids, axis=0)
    print('allids',allids.shape)
    allids = tf.tile(allids,[n_batch,1])
    
    #resort
    allids = tf.gather_nd(allids,ids)
    print('allids',allids.shape)
    
    #select in resorted space
    selids = make_batch_selection(outidx)
    selids=sprint(selids,'selids')
    outidx=sprint(outidx,'outidx')
    out_seed_ids = tf.gather_nd(allids,selids)
    
    return v,tf.cast(out_seed_ids,dtype=tf.int64)


def sparse_conv_prepare_2Dconv(sparse_dict,
                           space_dimensions):
    '''
    seed finding based on conv layers applied to a (learnable) distance matrix in 2 dimensions
    '''
    
    # assert len(conv_    kernels)==len(conv_filters)
    
    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                         sparse_dict['spatial_features_global'], \
                                                         sparse_dict['spatial_features_local'], \
                                                         sparse_dict['num_entries']
    
    n_vertices = colours_in.shape[1]
    next_square = (int(math.sqrt(float(int(n_vertices+1)))+1))**2
    xy_dim = int(math.sqrt(next_square))
    print(next_square)
    n_batch = colours_in.shape[0]
    n_features = colours_in.shape[2]
    
    use_features = tf.concat([space_global,colours_in,space_local], axis=-1)
    dimension_breakdown= tf.layers.dense(use_features, space_dimensions,activation=tf.nn.tanh,
                                         kernel_initializer=NoisyEyeInitializer)
    # break it down to 2 dimensions to get a 2D conv grid
    breakdown_x = (tf.layers.dense(dimension_breakdown, 1,activation=tf.nn.tanh)+3.)*2
    breakdown_y = tf.layers.dense(dimension_breakdown, 1,activation=tf.nn.tanh)
    sorting = breakdown_x+breakdown_y
    sorting = tf.squeeze(sorting) #only one dimension
    ids = get_sorted_vertices_ids(sorting)
    all_features = tf.concat([dimension_breakdown,(breakdown_x-3.)/2.,breakdown_y], axis=-1)

    sorted_all_features = tf.gather_nd(all_features,ids)
    #sorted_all_features= tf.squeeze(sorted_all_features)
    print('sorted_all_features',sorted_all_features.shape)
    
    padded = tf.pad(sorted_all_features,[[0,0],[0,next_square-n_vertices],[0,0]],"CONSTANT")
    x =  tf.reshape(padded,[n_batch,xy_dim,xy_dim,-1] )
    
    #FIXME: missing an indexing tensor to translate back to previous sorting !
    return x


    
def sparse_conv_seeded(sparse_dict, all_features_in, seed_indices, seed_scaling,nfilters, nspacefilters=1, 
                       nspacedim=3, nspacetransform=1,add_to_orig=True,
                       seed_talk=True,
                       name=None,
                       returnmerged=True):
    '''
    first nspacetransform uses just the untransformed first <nspacedim> entries of the space coordinates
    '''
    if name is None:
        name=""
    all_features=None 
    if sparse_dict is not None:
        colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                                        sparse_dict['spatial_features_global'], \
                                                                        sparse_dict['spatial_features_local'], \
                                                                        sparse_dict['num_entries'] 
        
        all_features = tf.concat([space_global,space_local,colours_in],axis=-1)   
    else:
        all_features = all_features_in



    trans_features = tf.layers.dense(all_features,nfilters,activation=tf.nn.relu)
    trans_features = tf.expand_dims(trans_features,axis=1)
    
    nbatch=all_features.shape[0]
    nvertex=all_features.shape[1]
    
    feature_layerout=[]
    space_layerout=[]
    seedselector = make_seed_selector(seed_indices)
    
    for i in range(nspacetransform):
        
        trans_space = all_features 
        trans_space = tf.layers.dense(trans_space/10.,nspacefilters,activation=tf.nn.tanh,
                                       kernel_initializer=NoisyEyeInitializer)

        trans_space = tf.layers.dense(trans_space*10.,nspacedim,activation=None,
                                      kernel_initializer=NoisyEyeInitializer, use_bias=False)
        trans_space = trans_space

        space_layerout.append(trans_space)
        
        seed_trans_space_orig = tf.gather_nd(trans_space,seedselector)
        seed_trans_space = tf.expand_dims(seed_trans_space_orig,axis=2)
        seed_trans_space = tf.tile(seed_trans_space,[1,1,nvertex,1])
        
        all_trans_space = tf.expand_dims(trans_space,axis=1)
        all_trans_space = tf.tile(all_trans_space,[1,seed_trans_space.shape[1],1,1])
        
        diff = all_trans_space - seed_trans_space
        
        diff = tf.reduce_sum(diff*diff,axis=-1)
        diff = normalise_distance_matrix(diff)
        
        
        diff = tf.expand_dims(diff,axis=3)
        
        thisout = diff*trans_features
        thisout = tf.reduce_sum(thisout,axis=2)
        
        #add back seed features
        seed_all_features = tf.gather_nd(all_features,seedselector)
        if seed_scaling is not None:
            seed_all_features = seed_scaling*seed_all_features
        
        #simple dense check this part
        #maybe add additional dense
        if seed_talk:
            #seed space transform?
            seed_distance = euclidean_squared(seed_trans_space_orig,seed_trans_space_orig)
            seed_distance = normalise_distance_matrix(seed_distance)
            seed_distance = tf.expand_dims(seed_distance,axis=3)
            seed_update = seed_distance*tf.expand_dims(seed_all_features,axis=1)
            seed_update = tf.reduce_sum(seed_update,axis=2)
            seed_merged_features = tf.concat([seed_all_features,seed_update],axis=-1)
            seed_all_features = tf.layers.dense(seed_merged_features,seed_all_features.shape[2],activation=tf.nn.relu,
                                                kernel_initializer=NoisyEyeInitializer)
    
        thisout = tf.concat([thisout,seed_all_features],axis=-1)
        #propagate back
        thisout = tf.expand_dims(thisout,axis=2)
        thisout = thisout*diff
        thisout = tf.transpose(thisout, perm=[0,2, 1,3])
        thisout = tf.reshape(thisout,[thisout.shape[0],thisout.shape[1],thisout.shape[2]*thisout.shape[3]])
        
        feature_layerout.append(thisout)
    
    feature_layerout = tf.concat(feature_layerout,axis=-1)
    space_layerout = tf.concat(space_layerout,axis=-1)
    
    
    #combien old features with new ones
    feature_layerout = tf.concat([all_features,space_layerout,feature_layerout,],axis=-1)
    feature_layerout = tf.layers.dense(feature_layerout/10.,nfilters, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)
    feature_layerout = feature_layerout*10.
    
    print('layer '+name+ ' feature_layerout out ', feature_layerout.shape)
    if returnmerged:
        return feature_layerout
    
    
    print('layer '+name+ ' space_global out ', space_global.shape)
    print('layer '+name+ ' space_layerout out ', space_layerout.shape)
    
    return construct_sparse_io_dict(feature_layerout , space_global, space_layerout, num_entries)
    
    
    
def sparse_conv_seedtalk(sparse_dict, seed_indices): 
     
    seedselector = make_seed_selector(seed_indices)
    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                                    sparse_dict['spatial_features_global'], \
                                                                    sparse_dict['spatial_features_local'], \
                                                                    sparse_dict['num_entries']
    seed_colours = tf.gather_nd(colours_in,seedselector)
    nfeatures = seed_colours.shape[2]
    nseeds = seed_colours.shape[1]
    seed_colours = tf.reshape(seed_colours,[seed_colours.shape[0],
                                            nseeds*nfeatures])
    print('seed_colours a',seed_colours.shape)
    #seed_colours = tf.squeeze(seed_colours, axis=-1)
    seed_colours = tf.layers.dense(seed_colours,nfeatures*2,activation=tf.nn.relu)
    seed_colours = tf.reshape(seed_colours,[seed_colours.shape[0],
                                            nseeds,nfeatures])
    print('seed_colours b',seed_colours.shape)
    
    #add back the seeds to all colours, inverse gather_nd
    layerout = colours_in
    layerout=tf.scatter_update(layerout, seedselector, seed_colours)
    print('layerout a',layerout.shape)
    
    return construct_sparse_io_dict(layerout , space_global, space_local, num_entries)

def sparse_conv_add_simple_seed_labels(net,seed_indices):
    colours_in, space_global, space_local, num_entries = net['all_features'], \
                                                                    net['spatial_features_global'], \
                                                                    net['spatial_features_local'], \
                                                                    net['num_entries']
    seedselector = make_seed_selector(seed_indices)
    seed_space = tf.gather_nd(space_global,seedselector)
    label = tf.argmin(euclidean_squared(space_global, seed_space), axis=-1)
    label = tf.cast(label,dtype=tf.float32)
    colours_in = tf.concat([colours_in,tf.expand_dims(label, axis=2)], axis=-1)
    return construct_sparse_io_dict(colours_in , space_global, space_local, num_entries)
    
def sparse_conv_batchnorm(net,momentum=0.9, training=True,**kwargs):
    colours_in, space_global, space_local, num_entries = net['all_features'], \
                                                                    net['spatial_features_global'], \
                                                                    net['spatial_features_local'], \
                                                                    net['num_entries']
    
    if momentum<=0:
        return net
    colours_in=tf.layers.batch_normalization(colours_in,training=training, momentum=momentum,**kwargs)
    space_global=tf.layers.batch_normalization(space_global,training=training, momentum=momentum,**kwargs)
    space_local=tf.layers.batch_normalization(space_local,momentum=momentum, training=training, **kwargs)
    
    return construct_sparse_io_dict(colours_in , space_global, space_local, num_entries)




