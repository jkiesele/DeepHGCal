import tensorflow as tf
from .neighbors import euclidean_squared,indexing_tensor, indexing_tensor_2, sort_last_dim_tensor, get_sorted_vertices_ids
from ops.nn import *
import numpy as np
from .initializers import NoisyEyeInitializer
from .activations import gauss_of_lin, gauss_times_linear, sinc, open_tanh, asymm_falling, gauss, multi_dim_edge_activation
import math

###helper functions
_sparse_conv_naming_index=0

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
    
    
    
def sparse_conv_collapse(sparse_dict):
    all_features, spatial_features_global, spatial_features_local, num_entries = sparse_dict['all_features'], \
                                                                                 sparse_dict['spatial_features_global'], \
                                                                                 sparse_dict['spatial_features_local'], \
                                                                                 sparse_dict['num_entries']
    return tf.concat([spatial_features_global,all_features , spatial_features_local],axis=-1)                                                                             
    

def zero_out_by_energy(net):
    return tf.cast(tf.not_equal(net[...,3],0), tf.float32)[..., tf.newaxis] * net


### maybe conv
def high_dim_dense(inputs,nodes,**kwargs):
    if len(inputs.shape) == 3:
        return tf.layers.conv1d(inputs, nodes, kernel_size=(1), strides=(1), padding='valid', 
                                **kwargs)
        
    if len(inputs.shape) == 4:
        return tf.layers.conv2d(inputs, nodes, kernel_size=(1,1), strides=(1,1), padding='valid', 
                                **kwargs)
        
    if len(inputs.shape) == 5:
        return tf.layers.conv3d(inputs, nodes, kernel_size=(1,1,1), strides=(1,1,1), padding='valid', 
                                **kwargs)



def sprint(tensor,pstr):
    #return tensor
    return tf.Print(tensor,[tensor],pstr,summarize=20)

def make_sequence(nfilters):
    isseq=(not hasattr(nfilters, "strip") and
            hasattr(nfilters, "__getitem__") or
            hasattr(nfilters, "__iter__"))
    
    if not isseq:
        nfilters=[nfilters]
    return nfilters

def sparse_conv_normalise(sparse_dict, log_energy=False):
    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                         sparse_dict['spatial_features_global'], \
                                                         sparse_dict['spatial_features_local'], \
                                                         sparse_dict['num_entries']
    
    scaled_colours_in = colours_in*1e-4
    if log_energy:
        scaled_colours_in = tf.log(colours_in+1)
    
    scaled_space_global=tf.concat([tf.expand_dims(space_global[:,:,0]/150.,axis=2),
                                   tf.expand_dims(space_global[:,:,1]/150.,axis=2),
                                   tf.expand_dims(space_global[:,:,2]/1600.,axis=2)],
                                  axis=-1)
    
    scaled_space_local = space_local/150.
    
    return construct_sparse_io_dict(scaled_colours_in, 
                                    scaled_space_global, 
                                    scaled_space_local, 
                                    num_entries)

#just for compat
def make_batch_selection(ids):
    n_batch=ids.shape[0]
    n_vertices=ids.shape[1]
    ids = tf.cast(ids, dtype=tf.int64)
    batch=tf.range(n_batch, dtype=tf.int64)
    batch = tf.tile(batch[..., tf.newaxis, tf.newaxis], [1,n_vertices,1])
    select = tf.concat((batch, ids[..., tf.newaxis]), axis=-1)
    return select
    

def select_based_on(vertices_to_select_from, select_criterion_max, n_select): 
       
    _, I = tf.nn.top_k(select_criterion_max, n_select)
    selection = make_batch_selection(I)
    return tf.gather_nd(vertices_to_select_from,selection)
    

def apply_distance_weight(x, zero_is_one=False):
    #return x
    if zero_is_one:
        return gauss_of_lin(x)
    else:
        return gauss_times_linear(x)
    
def get_rot_symmetric_distance(raw_difference):
    rot_raw_difference = tf.reduce_sum(raw_difference*raw_difference,axis=-1)
    rot_raw_difference = tf.sqrt(rot_raw_difference+1e-6)
    rot_raw_difference = tf.expand_dims(rot_raw_difference,axis=3)
    return rot_raw_difference
    
def add_rot_symmetric_distance(raw_difference):
    
    edges = tf.concat([get_rot_symmetric_distance(raw_difference),raw_difference],axis=-1)
    return edges
    

def create_edges(vertices_a, vertices_b, zero_is_one_weight=False, n_properties=-1, norotation=False,
                 apply_activation=True):
    #BxVxF
    expanded_vertices_a = tf.expand_dims(vertices_a, axis=1)
    expanded_vertices_b = tf.expand_dims(vertices_b, axis=2)
    raw_difference = expanded_vertices_a - expanded_vertices_b # Bx1xVxF - BxVx1xF = B x V x V x F
    #calculate explicitly rotational symmetric on
    edges = None
    if not norotation:
        edges = add_rot_symmetric_distance(raw_difference) # BxVxVx(F+1)
    else: 
        edges = raw_difference
        
    if n_properties>0:
        edges = edges[:,:,:,0:n_properties]
        
    
    
    if apply_activation:   
        return apply_distance_weight(edges,zero_is_one_weight)
    else:
        return edges
    
def sparse_conv_multipl_dense(in_tensor, nfilters, activation=None, kernel_initializer=tf.glorot_normal_initializer, name=None):
    global _sparse_conv_naming_index
    _sparse_conv_naming_index+=1
    if name is None:
        name='sparse_conv_multipl_dense_'+str(_sparse_conv_naming_index)
    
    broadcast_shape = [1 for i in range(len(in_tensor.shape)-1)]
    batch_etc = [int(in_tensor.shape[i]) for i in range(len(in_tensor.shape)-1)]
    print('batch_etc',batch_etc)
    weights_shape = broadcast_shape + [nfilters, int(in_tensor.shape[-1])]
    bias_shape = broadcast_shape + [nfilters]
    
    print('in_tensor.shape', in_tensor.shape)
    weights = tf.get_variable(name+"_weights", [nfilters,int(in_tensor.shape[-1])],dtype=tf.float32,
                                        initializer=kernel_initializer())/10.

    weights = tf.reshape(weights, weights_shape)
    
    print('weights',weights.shape)
    
    bias = tf.get_variable(name+"_bias", [nfilters],dtype=tf.float32,
                                        initializer=tf.zeros_initializer())    
    
    expanded_in_tensor = tf.expand_dims(in_tensor, axis=-2)
    
    print('weights',weights.shape)
    
    applied_weights = weights*expanded_in_tensor + 1.
    
    print('applied_weights',applied_weights.shape)
    
    collapsed = tf.reduce_prod(applied_weights, axis=-1) - 1.
    
    collapsed = tf.nn.bias_add(collapsed, bias)
    #exit()
    if activation is not None:
        return activation(collapsed)
    else:
        return collapsed
    
    
    
def create_active_edges(vertices_a, vertices_b, name,multiplier=1):
    '''
    learnable:
    -global scaler for receptive field in Y with Y: BxYxVxF
    -local frequency a for activation: exp(x^2)cos(a x)
    '''
    
    expanded_vertices_a = tf.expand_dims(vertices_a, axis=1)
    expanded_vertices_b = tf.expand_dims(vertices_b, axis=2)
    edges = expanded_vertices_a - expanded_vertices_b
    for i in range(multiplier-1):
        edges = tf.concat([edges,edges],axis=-1)
    
    rec_field_scaler = tf.get_variable(name+"_rec_field_scaler", [edges.shape[1]],dtype=tf.float32,
                                        initializer=tf.zeros_initializer)
    rec_field_scaler = tf.expand_dims(tf.expand_dims(tf.expand_dims(rec_field_scaler,axis=0),axis=2),axis=3)
    edges = edges * (rec_field_scaler+0.1 )#+ 1.)
    
    frequency_scaler = tf.get_variable(name+"_frequency_scaler", [edges.shape[-1]],dtype=tf.float32)
    frequency_scaler = tf.expand_dims(tf.expand_dims(tf.expand_dims(frequency_scaler,axis=0),axis=0),axis=0)
    
    edges = tf.exp(-edges*edges)*tf.cos(frequency_scaler*5.* edges)
    print('create_active_edges: edges out ',name,edges.shape)
    return edges
    
    
def apply_edges(vertices, edges, reduce_sum=True, flatten=True,expand_first_vertex_dim=True, aggregation_function=tf.reduce_max): 
    '''
    edges are naturally BxVxV'xF
    vertices are BxVxF'  or BxV'xF'
    This function returns BxVxF'' if flattened and summed
    '''
    edges = tf.expand_dims(edges,axis=3)
    if expand_first_vertex_dim:
        vertices = tf.expand_dims(vertices,axis=1)
    vertices = tf.expand_dims(vertices,axis=4)

    out = edges*vertices # [BxVxV'x1xF] x [Bx1xV'xF'x1] = [BxVxV'xFxF']

    if reduce_sum:
        out = aggregation_function(out,axis=2)
    if flatten:
        out = tf.reshape(out,shape=[out.shape[0],out.shape[1],-1])
    
    return out

 
def apply_space_transform(vertices, units_transform, output_dimensions,
                          depth=1,activation=open_tanh): 
    trans_space = vertices
    for i in range(depth):
        trans_space = tf.layers.dense(trans_space,units_transform,activation=activation,
                                       kernel_initializer=NoisyEyeInitializer)
        trans_space = trans_space
    trans_space = tf.layers.dense(trans_space,output_dimensions,activation=None,
                                  kernel_initializer=NoisyEyeInitializer, use_bias=True)
    return trans_space
########

def sparse_conv_add_simple_seed_labels(net,seed_indices):
    colours_in, space_global, space_local, num_entries = net['all_features'], \
                                                                    net['spatial_features_global'], \
                                                                    net['spatial_features_local'], \
                                                                    net['num_entries']
    seedselector = make_batch_selection(seed_indices)
    seed_space = tf.gather_nd(space_global,seedselector)
    label = tf.argmin(euclidean_squared(space_global, seed_space), axis=-1)
    label = tf.cast(label,dtype=tf.float32)
    colours_in = tf.concat([colours_in,tf.expand_dims(label, axis=2)], axis=-1)
    return construct_sparse_io_dict(colours_in , space_global, space_local, num_entries)
    

def get_distance_weight_to_seeds(vertices_in, seed_idx, dimensions=4, add_zeros=0):
    
    seedselector = make_batch_selection(seed_idx)
    vertices_in = tf.layers.dense (vertices_in, dimensions, kernel_initializer=NoisyEyeInitializer)
    seed_vertices = tf.gather_nd(vertices_in,seedselector)
    edges = create_edges(vertices_in,seed_vertices, zero_is_one_weight=True)
    distance = edges[:,:,:,0]
    distance = tf.transpose(distance, perm=[0,2,1])
    if add_zeros>0:
        zeros = tf.zeros_like(distance[:,:,0], dtype=tf.float32)
        zeros = tf.expand_dims(zeros, axis=2)
        for i in range(add_zeros):
            distance = tf.concat([distance,zeros], axis=-1)
    return distance
    
    

def sparse_conv_seeded3(vertices_in, 
                       seed_indices, 
                       nfilters, 
                       nspacefilters=32, 
                       nspacedim=3, 
                       seed_talk=True,
                       compress_before_propagate=True,
                       use_edge_properties=-1):
    global _sparse_conv_naming_index
    '''
    '''
    #for later
    _sparse_conv_naming_index+=1
    
    seedselector = make_batch_selection(seed_indices) # To select seeds from all the features using gather_nd
    
    trans_space = apply_space_transform(vertices_in, nspacefilters,nspacedim) # Just a couple of dense layers

    seed_trans_space = tf.gather_nd(trans_space,seedselector) # Select seeds from transformed space

    edges = create_edges(trans_space,seed_trans_space,n_properties=use_edge_properties) # BxVxV'xF

    trans_vertices = tf.layers.dense(vertices_in,nfilters,activation=tf.nn.relu) # Just dense again

    expanded_collapsed = apply_edges(trans_vertices, edges, reduce_sum=True, flatten=True) # [BxVxF]
   
    #add back seed features
    seed_all_features = tf.gather_nd(trans_vertices,seedselector)
    
    #simple dense check this part
    #maybe add additional dense
    if seed_talk:
        #seed space transform?
        seed_edges = create_edges(seed_trans_space,seed_trans_space,n_properties=use_edge_properties)
        trans_seeds = apply_edges(seed_all_features, seed_edges, reduce_sum=True, flatten=True)
        seed_merged_features = tf.concat([seed_all_features,trans_seeds],axis=-1)
        seed_all_features = tf.layers.dense(seed_merged_features,seed_all_features.shape[2],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=NoisyEyeInitializer)
    
    #compress
    
    expanded_collapsed = tf.concat([expanded_collapsed,seed_all_features],axis=-1)
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,nfilters, activation=tf.nn.tanh)
        
    print('expanded_collapsed',expanded_collapsed.shape)
    
    #propagate back, transposing the edges does the trick, now they point from Nseeds to Nvertices
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    expanded_collapsed = apply_edges(expanded_collapsed, edges, reduce_sum=False, flatten=True)
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,nfilters, activation=tf.nn.tanh,
                                             kernel_initializer=NoisyEyeInitializer)
    

    #combien old features with new ones
    feature_layerout = tf.concat([vertices_in,trans_space,expanded_collapsed],axis=-1)
    feature_layerout = tf.layers.dense(feature_layerout,nfilters,activation=tf.nn.tanh,
                                       kernel_initializer=NoisyEyeInitializer)
    return feature_layerout

def sparse_conv_global_exchange(vertices_in, 
                                aggregate_function=tf.reduce_mean,
                                expand_to_dims=-1,
                                collapse_to_dims=-1,
                                learn_global_node_placement_dimensions=None):
    
    trans_vertices_in = vertices_in
    if expand_to_dims>0:
        trans_vertices_in = tf.layers.dense(trans_vertices_in,expand_to_dims,activation=tf.nn.relu)
    
    global_summed = None    
    if learn_global_node_placement_dimensions is not None:
        trans_vertices_in_space = trans_vertices_in[:,:,0:learn_global_node_placement_dimensions]
        global_node_placement = tf.reduce_mean(trans_vertices_in_space,axis=1, keepdims=True)
        edges = create_edges(trans_vertices_in_space, global_node_placement,
                             norotation=True)
        edges = gauss_times_linear(edges)
        global_summed = apply_edges(trans_vertices_in, edges, reduce_sum=True, flatten=True)
        
        
    else: 
        global_summed = tf.reduce_mean(trans_vertices_in, axis=1, keepdims=True)
        
    global_summed = tf.tile(global_summed,[1,vertices_in.shape[1],1])
    vertices_out = tf.concat([vertices_in,global_summed],axis=-1)
    if collapse_to_dims>0:
        vertices_out = tf.layers.dense(vertices_out, collapse_to_dims, activation=tf.nn.relu)
    
    return vertices_out
    

    
def sparse_conv_moving_seeds2(vertices_in, 
                             n_filters, 
                             n_seeds, 
                             n_seed_dimensions,
                             use_edge_properties=-1,
                             n_spacefilters=64,
                             seed_filters=[64],
                             compress_before_propagate=False,
                             seed_talk=True,
                             seed_positions=None,
                             edge_multiplicity=1,
                             add_back_original=True):
    
    global _sparse_conv_naming_index
    _sparse_conv_naming_index += 1
    this_name = "sparse_conv_moving_seeds2_"+str(_sparse_conv_naming_index)
    print(this_name)
    
    trans_global_vertices = apply_space_transform(vertices_in, n_spacefilters,n_seed_dimensions) # Just a couple of dense layers
   
    trans_vertices = tf.layers.dense(vertices_in,n_filters,activation=tf.nn.relu)
    
    highest_activation_vertices = select_based_on(trans_global_vertices, 
                                                  tf.reduce_max(vertices_in,axis=-1),
                                                  n_seeds)
    highest_activation_vertices = tf.reshape(highest_activation_vertices, 
                                             [highest_activation_vertices.shape[0], -1])
    
    seed_input = tf.concat([highest_activation_vertices,tf.reduce_mean(trans_global_vertices,axis=1)],
                           axis=-1)
    
    if seed_positions is not None:
        seed_input = tf.concat([seed_input, tf.reshape(seed_positions, [seed_positions.shape[0],-1])],axis=-1)
    
    for f in seed_filters:
        seed_input = tf.layers.dense(seed_input,f,activation=open_tanh,kernel_initializer=NoisyEyeInitializer)
    
    seed_input = tf.layers.dense(seed_input,n_seeds*n_seed_dimensions,activation=open_tanh,
                                 kernel_initializer=NoisyEyeInitializer,
                                 bias_initializer=tf.random_normal_initializer(0., stddev=1))
    seed_positions = tf.reshape(seed_input,[seed_input.shape[0],n_seeds,n_seed_dimensions])
    
    print('seed_positions',seed_positions.shape)
    
    edges = create_active_edges(trans_global_vertices,seed_positions,multiplier=edge_multiplicity,name=this_name)
    
    print('sparse_conv_moving_seeds: edges ', edges.shape)
    expanded_collapsed = apply_edges(trans_vertices, edges, reduce_sum=True, flatten=True)
    
    
    
    print('expanded_collapsed before seed talk',expanded_collapsed.shape)
    #simple dense check this part
    #maybe add additional dense
    if seed_talk:
        #seed space transform?
        seed_features = tf.layers.dense(expanded_collapsed,n_filters,activation=tf.nn.relu)
        seed_edges = create_edges(seed_positions,seed_positions,n_properties=use_edge_properties)
        trans_seeds = apply_edges(seed_features, seed_edges, reduce_sum=True, flatten=True)
        seed_merged_features = tf.concat([expanded_collapsed,trans_seeds],axis=-1)
        seed_all_features = tf.layers.dense(seed_merged_features,n_filters,
                                            activation=tf.nn.tanh,
                                            kernel_initializer=NoisyEyeInitializer)
        expanded_collapsed = tf.concat([expanded_collapsed,seed_all_features],axis=-1)
        print('expanded_collapsed after seed talk',expanded_collapsed.shape)
        
        
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters,activation=tf.nn.relu)
        
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    expanded_collapsed = apply_edges(expanded_collapsed, edges, reduce_sum=False, flatten=True)
    
    if compress_before_propagate or not add_back_original:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,
                                             kernel_initializer=NoisyEyeInitializer)
    if not add_back_original:
        return expanded_collapsed, seed_positions
    
    expanded_collapsed = tf.concat([vertices_in,expanded_collapsed],axis=-1)
    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,
                                         kernel_initializer=NoisyEyeInitializer)
    print('sparse_conv_moving_seeds out',expanded_collapsed.shape)
    return expanded_collapsed, seed_positions
    


def sparse_conv_make_neighbors2(vertices_in, num_neighbors=10, 
                               output_all=15, space_transformations=10,
                               merge_neighbours=1,
                               edge_activation=gauss_of_lin,
                               indexing=None,
                               edge_transformations=None,
                               train_space=True,
                               train_global_space=False,
                               ):
    
    assert merge_neighbours <= num_neighbors
    global _sparse_conv_naming_index
    space_initializer = None #NoisyEyeInitializer
    
    #for later
    _sparse_conv_naming_index+=1
    
    space_transformations = make_sequence(space_transformations)
    output_all = make_sequence(output_all)
    
    
    trans_space = vertices_in
    
    if train_global_space:
        trans_space = trans_space[0:1,:,:]
        trans_space = tf.concat([trans_space[:,:,0:3],trans_space[:,:,4:]],axis=-1)
    if train_space:
        for i in range(len(space_transformations)):
            if i< len(space_transformations)-1:
                trans_space = tf.layers.dense(trans_space/10.,space_transformations[i],activation=open_tanh,
                                           kernel_initializer=space_initializer)
                trans_space*=10.
            else:
                trans_space = tf.layers.dense(trans_space/10.,space_transformations[i],activation=open_tanh,
                                           kernel_initializer=space_initializer)
                trans_space*=10.
                
    else:
        trans_space = vertices_in[:,:,0:space_transformations[-1]]

    indexing = None
    if train_global_space:
        indexing, _ = indexing_tensor_2(trans_space[0:1,:,:], num_neighbors, n_batch=trans_space.shape[0])
        trans_space = tf.tile(trans_space,[vertices_in.shape[0],1,1])
    else:
        indexing, _ = indexing_tensor_2(trans_space, num_neighbors)
    
    neighbour_space = tf.gather_nd(trans_space, indexing)
    
    #build edges manually
    expanded_trans_space = tf.expand_dims(trans_space, axis=2)
    diff = expanded_trans_space - neighbour_space
    edges = add_rot_symmetric_distance(diff)
    #edges = apply_distance_weight(edges)
    edges = tf.expand_dims(edges,axis=3)
    
    if edge_transformations is not None:
        edge_transformations = make_sequence(edge_transformations)
        assert len(edge_transformations) == len(output_all)
        
    updated_vertices = vertices_in
    orig_edges = edges
    for i in range(len(output_all)):
        f = output_all[i]
        #interpret distances in a different way -> dense on edges (with funny activations TBI)
        if f < 0:
            #this is a global interaction
            global_summed = tf.reduce_mean(updated_vertices, axis=1, keepdims=True)
            global_summed = tf.tile(global_summed,[1,updated_vertices.shape[1],1])
            updated_vertices = tf.concat([updated_vertices,global_summed],axis=-1)
            continue
        
        
        concat_edges = tf.concat([orig_edges,edges],axis=-1)
        if edge_transformations is not None:
            concat_edges = tf.layers.dense(concat_edges, edge_transformations[i],
                                           activation=edge_activation)
            
        edges = tf.layers.dense(concat_edges, 
                                edges.shape[-1],activation=edge_activation,
                                kernel_initializer = space_initializer)
        
        vertex_with_neighbours = tf.gather_nd(updated_vertices, indexing)
        vertex_with_neighbours = tf.expand_dims(vertex_with_neighbours,axis=4)
        flattened_gathered = vertex_with_neighbours * edges
        flattened_gathered = tf.reduce_mean(flattened_gathered, axis=2)
        flattened_gathered = tf.reshape(flattened_gathered, shape=[flattened_gathered.shape[0],
                                                                   flattened_gathered.shape[1],-1])
        updated_vertices = tf.layers.dense(tf.concat([vertices_in,flattened_gathered],axis=-1), 
                                           f, activation=tf.nn.relu) 


    updated_vertices = tf.concat([trans_space,updated_vertices],axis=-1)
        
    return updated_vertices



def sparse_conv_edge_conv(vertices_in, num_neighbors=30, 
                      mpl_layers=[64,64,64],
                      aggregation_function = tf.reduce_max,
                      share_keyword=None, #TBI,
                      edge_activation=None
                      ):
    
    trans_space = vertices_in
    indexing, _ = indexing_tensor_2(trans_space, num_neighbors)
    #change indexing to be not self-referential
    neighbour_space = tf.gather_nd(vertices_in, indexing)
    
    expanded_trans_space = tf.expand_dims(trans_space, axis=2)
    expanded_trans_space = tf.tile(expanded_trans_space,[1,1,num_neighbors,1])
    
    diff = expanded_trans_space - neighbour_space
    edge = tf.concat([expanded_trans_space,diff], axis=-1)
    
    for f in mpl_layers:
        edge = tf.layers.dense(edge, f, activation=tf.nn.relu)
        
    if edge_activation is not None:
        edge = edge_activation(edge)
        
    vertex_out = aggregation_function(edge,axis=2)
    
    return vertex_out
    


def max_pool_on_last_dimensions(vertices_in, n_output_vertices):
    
    all_features = vertices_in
    
    _, I = tf.nn.top_k(tf.reduce_max(all_features, axis=2), n_output_vertices)
    I = tf.expand_dims(I, axis=2)

    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, vertices_in.shape[0]), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1,n_output_vertices, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)
    
    return tf.gather_nd(vertices_in, _indexing_tensor)
    

def create_active_edges2(vertices_a, vertices_b, name,multiplier=1,skew_and_var=None,fixed_frequency=-1,
                         no_activation=False, return_rot=False,
                         fixed_field=False):
    '''
    learnable:
    -global scaler for receptive field in Y with Y: BxYxVxF
    -local frequency a for activation: exp(x^2)cos(a x)
    if skew_and_var is given, anything found there is  input to the receptive field and
    frquency calculation. needs to be same shape as vertices_b
    '''
    
    expanded_vertices_a = tf.expand_dims(vertices_a, axis=1)
    expanded_vertices_b = tf.expand_dims(vertices_b, axis=2)
    edges = expanded_vertices_a - expanded_vertices_b
    for i in range(multiplier-1):
        edges = tf.concat([edges,edges],axis=-1)
    
    n_parameters = edges.shape[1]*edges.shape[-1]
    
    if skew_and_var is None:
    
        rec_field_scaler = tf.get_variable(name+"_rec_field_scaler", [n_parameters],dtype=tf.float32,
                                            initializer=tf.zeros_initializer) + 1.
        frequency_scaler = tf.get_variable(name+"_frequency_scaler", [n_parameters],dtype=tf.float32)
        rec_field_scaler = tf.reshape(rec_field_scaler,[1,edges.shape[1],1,edges.shape[-1]])    
        frequency_scaler = tf.reshape(frequency_scaler,[1,edges.shape[1],1,edges.shape[-1]]) 
        
        
    else:
        assert vertices_b.shape[1]==skew_and_var.shape[1]
        rec_field_scaler=[]
        frequency_scaler=[]
        for i in range(int(edges.shape[1])):
            rec_field_scaler.append(tf.expand_dims(tf.layers.dense(skew_and_var[:,i,:], edges.shape[-1], activation=tf.nn.relu),axis=1))
            frequency_scaler.append(tf.expand_dims(tf.layers.dense(skew_and_var[:,i,:], edges.shape[-1], activation=tf.nn.relu,
                                                                   bias_initializer=tf.random_uniform_initializer(0., 10.),
                                                                   kernel_initializer=tf.random_normal_initializer(0., 0.01)),axis=1))
        
        rec_field_scaler = tf.expand_dims(tf.concat(rec_field_scaler,axis=1),axis=2)/10.+1.
        frequency_scaler = tf.expand_dims(tf.concat(frequency_scaler, axis=1),axis=2)
       
    rot_symm = get_rot_symmetric_distance(edges)
    
    if fixed_field:
        rec_field_scaler=1.
        
    if no_activation:
        if return_rot:
            return rec_field_scaler*edges, rot_symm
        else:
            return rec_field_scaler*edges
    
        
    elif fixed_frequency<0:
        edges = tf.exp(-(rec_field_scaler*edges)**2.)*tf.cos(frequency_scaler* edges)
    else:
        edges = tf.exp(-(rec_field_scaler*edges)**2.)*tf.cos(fixed_frequency* edges)
    
    return edges
    
    
def sparse_conv_moving_seeds3(vertices_in, 
                             n_filters, 
                             n_seeds, 
                             n_seed_dimensions,
                             seed_filters=[],
                             compress_before_propagate=False,
                             edge_multiplicity=1):
    
    global _sparse_conv_naming_index
    _sparse_conv_naming_index += 1
    this_name = "sparse_conv_moving_seeds3_"+str(_sparse_conv_naming_index)
    
    transformed_vertex_positions = tf.layers.dense(vertices_in, n_seed_dimensions, activation=tf.nn.tanh,
                                    kernel_initializer=NoisyEyeInitializer)
    seed_positions=[]
    for i in range(n_seeds):
        positions = vertices_in
        for f in seed_filters:
            positions = tf.layers.dense(positions, n_seed_dimensions, activation=tf.nn.tanh)
        positions = tf.layers.dense(positions, n_seed_dimensions, activation=tf.nn.tanh,
                                    kernel_initializer=NoisyEyeInitializer)
        weights = tf.layers.dense(vertices_in, 1, activation=tf.nn.tanh,
                                  kernel_initializer=tf.random_normal_initializer(0, 0.01))+1.
        weighted_positions = positions * weights
        seed_position = tf.reduce_sum(weighted_positions,axis=1, keepdims=True)/tf.reduce_sum(weights,axis=1,keepdims=True)
        seed_positions.append(seed_position)
    
    seed_positions = tf.concat(seed_positions, axis=1)
    
    edges = create_active_edges2(transformed_vertex_positions,seed_positions,
                                multiplier=edge_multiplicity,name=this_name)
    
    expanded_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True)
    
    #if seed_talk:
    ##simple seed talk
    #    expanded_collapsed = tf.reshape(expanded_collapsed, [expanded_collapsed.shape[0],
    #                                                         1,
    #                                                         expanded_collapsed.shape[1]
    #                                                         *expanded_collapsed.shape[2]])
    #    #if compress_before_propagate:
    #    #    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters,activation=tf.nn.relu)
    #        
    #    expanded_collapsed = tf.layers.dense(expanded_collapsed,expanded_collapsed.shape[-1],activation=tf.nn.relu,
    #                                         kernel_initializer=NoisyEyeInitializer)
    #    
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters,activation=tf.nn.relu)
        
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    expanded_collapsed = apply_edges(expanded_collapsed, edges, reduce_sum=False, flatten=True)
    
    
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters,activation=tf.nn.relu)
        
    
    expanded_collapsed = tf.concat([vertices_in,expanded_collapsed],axis=-1)
    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,
                                         kernel_initializer=NoisyEyeInitializer)
    print('sparse_conv_moving_seeds3 out',expanded_collapsed.shape)
    return expanded_collapsed, seed_positions

    
    
    
    
def sparse_conv_moving_seeds4(vertices_in, 
                             n_filters, 
                             n_propagate,
                             n_seeds, 
                             n_seed_dimensions,
                             weight_filters=[],
                             seed_filters=[],
                             out_filters=[],
                             edge_multiplicity=1,
                             add_egde_info=False):
    
    global _sparse_conv_naming_index
    _sparse_conv_naming_index += 1
    this_name = "sparse_conv_moving_seeds3_"+str(_sparse_conv_naming_index)
    
    transformed_vertex_positions = tf.layers.dense(vertices_in, n_seed_dimensions, activation=tf.nn.tanh,
                                    kernel_initializer=NoisyEyeInitializer)
    
    propagate_features =  tf.layers.dense(vertices_in, n_propagate, activation=tf.nn.tanh)
    
    seed_positions=[] 
    skew_and_var=[]
    for i in range(n_seeds):
        weight_input = vertices_in
        for f in weight_filters:
            weight_input = tf.layers.dense(weight_input, f, activation=tf.nn.relu)
        
        weights = tf.layers.dense(vertices_in, 1, activation=tf.nn.relu)+1e-4
        weight_sum = tf.reduce_sum(weights,axis=1,keepdims=True)
        
        max_position = select_based_on(transformed_vertex_positions, tf.reshape(weights, [weights.shape[0],weights.shape[1]]), n_select=1)
        min_position = select_based_on(transformed_vertex_positions, -tf.reshape(weights, [weights.shape[0],weights.shape[1]]), n_select=1)
        mean_position = tf.reduce_sum(transformed_vertex_positions * weights,axis=1, keepdims=True)/weight_sum
        diff_to_mean = transformed_vertex_positions-mean_position
        pos_sq = diff_to_mean **2
        var_position  = tf.reduce_sum(pos_sq * weights,axis=1, keepdims=True)/weight_sum
        skew_position  = tf.reduce_sum(pos_sq * diff_to_mean * weights,axis=1, keepdims=True)/weight_sum
        
        seed_position = tf.concat([max_position,min_position,mean_position,var_position,skew_position],axis=-1)
        seed_position = tf.layers.dense(seed_position,n_seed_dimensions,activation=tf.nn.tanh)
        
        seed_positions.append(seed_position)
        skew_and_var.append(tf.concat([mean_position,var_position,skew_position],axis=-1))
    
    seed_positions = tf.concat(seed_positions, axis=1)
    skew_and_var = tf.concat(skew_and_var, axis=1)
    
    edges, rot_edge = create_active_edges2(transformed_vertex_positions,seed_positions,
                                multiplier=edge_multiplicity,name=this_name,
                                skew_and_var=None,fixed_frequency=0,return_rot=True,no_activation=True)
    
    collapse_edges = tf.concat([ gauss_of_lin(rot_edge), gauss_times_linear(edges)],axis=-1)
    expand_edges = gauss_of_lin(rot_edge)
    
    expanded_collapsed = apply_edges(propagate_features, collapse_edges, reduce_sum=True, flatten=True)
      
    
    for f in seed_filters:
        expanded_collapsed = tf.layers.dense(expanded_collapsed, f, activation=tf.nn.relu)
        
    expand_edges = tf.transpose(expand_edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    expanded_collapsed = apply_edges(expanded_collapsed, expand_edges, reduce_sum=False, flatten=True)
    
    #also add back the distance to the aggregator 'aka edges' as information
    #and make this part a bit deeper.  
    if add_egde_info:
        collapse_edges = tf.transpose(collapse_edges, perm=[0,2, 1,3])
        collapse_edges = tf.reshape(collapse_edges, [collapse_edges.shape[0], collapse_edges.shape[1],-1])
        expanded_collapsed = tf.concat([collapse_edges,expanded_collapsed],axis=-1)
    
    expanded_collapsed = tf.concat([vertices_in,expanded_collapsed],axis=-1)

    for f in out_filters:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,f,activation=tf.nn.relu,kernel_initializer=NoisyEyeInitializer)
        
    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)
    
    print('sparse_conv_moving_seeds4 out',expanded_collapsed.shape)
    return expanded_collapsed, seed_positions

    
    
    
def sparse_conv_moving_seeds6(vertices_in, 
                             n_filters, 
                             n_propagate,
                             n_seeds, 
                             n_seed_dimensions,
                             weight_filters=[],
                             seed_filters=[],
                             out_filters=[],
                             edge_multiplicity=1,
                             add_egde_info=False):
    
    global _sparse_conv_naming_index
    _sparse_conv_naming_index += 1
    this_name = "sparse_conv_moving_seeds3_"+str(_sparse_conv_naming_index)
    
    transformed_vertex_positions = tf.layers.dense(vertices_in, n_seed_dimensions, activation=tf.nn.tanh,
                                    kernel_initializer=NoisyEyeInitializer)
    pass_through_vertex_features = tf.layers.dense(vertices_in, n_seed_dimensions+n_propagate, activation=tf.nn.tanh,
                                                   kernel_initializer=None)
    propagate_features =  []

    seed_positions=[] 
    
    skew_and_var=[]
    for i in range(n_seeds):
        prop_feat = tf.expand_dims(tf.layers.dense(vertices_in, n_propagate, activation=tf.nn.relu),axis=1)
        propagate_features.append(prop_feat)
        
        weights = tf.layers.dense(vertices_in, 1, activation=tf.nn.relu)+1e-4
        weight_sum = tf.reduce_sum(weights,axis=1,keepdims=True)
        
        max_position = select_based_on(transformed_vertex_positions, tf.reshape(weights, [weights.shape[0],weights.shape[1]]), n_select=1)
        mean_position = tf.reduce_sum(transformed_vertex_positions * weights,axis=1, keepdims=True)/weight_sum
        diff_to_mean = transformed_vertex_positions-mean_position
        pos_sq = diff_to_mean **2
        var_position  = tf.reduce_sum(pos_sq * weights,axis=1, keepdims=True)/weight_sum
        skew_position  = tf.reduce_sum(pos_sq * diff_to_mean * weights,axis=1, keepdims=True)/weight_sum
        
        seed_position = tf.concat([max_position,mean_position,var_position,skew_position],axis=-1)
        seed_position = tf.layers.dense(seed_position,n_seed_dimensions,activation=tf.nn.tanh)
        
        seed_positions.append(seed_position)
        skew_and_var.append(tf.concat([mean_position,var_position,skew_position],axis=-1))
    
    propagate_features = tf.concat(propagate_features,axis=1)
    seed_positions = tf.concat(seed_positions, axis=1)
    skew_and_var = tf.concat(skew_and_var, axis=1)
    
    edges, rot_edge = create_active_edges2(transformed_vertex_positions,seed_positions,
                                multiplier=edge_multiplicity,name=this_name,
                                skew_and_var=None,fixed_frequency=0,
                                return_rot=True,no_activation=True)
    
    collapse_edges = gauss_of_lin(edges)
    expand_edges = gauss_of_lin(rot_edge)
    
    expanded_collapsed = apply_edges(propagate_features, collapse_edges, 
                                     reduce_sum=True, flatten=True, 
                                     expand_first_vertex_dim=False)

    expand_edges = tf.transpose(expand_edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    expanded_collapsed = apply_edges(expanded_collapsed, expand_edges, reduce_sum=False, flatten=False)
    
    print('expanded_collapsed',expanded_collapsed.shape)
    expanded_collapsed = tf.reduce_max(expanded_collapsed, axis=2)
    expanded_collapsed = tf.reshape(expanded_collapsed,[expanded_collapsed.shape[0],
                                                        expanded_collapsed.shape[1],-1])
    
    
    expanded_collapsed = tf.concat([pass_through_vertex_features,expanded_collapsed],axis=-1)
        
    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)
    
    print('sparse_conv_moving_seeds5 out',expanded_collapsed.shape)
    return expanded_collapsed, seed_positions

    
    
    
    

    
def half_nodes(x): 
    nodes = int(int(x.shape[-1])/2)
    if nodes<1:
        nodes = 1  
    return  tf.layers.dense(x, nodes , activation=tf.nn.relu)
    

    
def sparse_conv_moving_seeds5(vertices_in, 
                             n_filters, 
                             n_propagate,
                             n_seeds, 
                             n_seed_dimensions,
                             weight_filters=[],
                             edge_multiplicity=1,
                             add_egde_info=False):   
    
    global _sparse_conv_naming_index
    _sparse_conv_naming_index += 1
    this_name = "sparse_conv_moving_seeds6_"+str(_sparse_conv_naming_index)
    
    transformed_vertex_positions = tf.layers.dense(vertices_in, n_seed_dimensions, activation=tf.nn.relu,
                                    kernel_initializer=NoisyEyeInitializer)
    
    propagate_features =  tf.layers.dense(vertices_in, n_propagate, activation=tf.nn.relu)
    
    seed_positions=[] 
    skew_and_var=[]
    for i in range(n_seeds):
        weight_input = vertices_in
        for f in weight_filters:
            weight_input = tf.layers.dense(weight_input, f, activation=tf.nn.relu)
        
        weights = tf.layers.dense(vertices_in, 1, activation=tf.nn.relu)+1e-4
        print('weights',weights.shape)
        weight_sum = tf.reduce_sum(weights,axis=1,keepdims=True)
        
        max_position = select_based_on(transformed_vertex_positions, 
                                       tf.reshape(weights, [weights.shape[0],weights.shape[1]]), 
                                       n_select=1)
        mean_position = tf.reduce_sum(transformed_vertex_positions * weights,axis=1, keepdims=True)/weight_sum
        diff_to_mean = transformed_vertex_positions-mean_position
        pos_sq = diff_to_mean **2
        var_position  = tf.reduce_sum(pos_sq * weights,axis=1, keepdims=True)/weight_sum
        skew_position  = tf.reduce_sum(pos_sq * diff_to_mean * weights,axis=1, keepdims=True)/weight_sum
        
        seed_position = tf.concat([max_position,mean_position,var_position,skew_position],axis=-1)
        seed_position = tf.layers.dense(seed_position,seed_position.shape[-1]*2,activation=tf.nn.relu)
        seed_position = tf.layers.dense(seed_position,seed_position.shape[-1],activation=tf.nn.relu)
        seed_position = tf.layers.dense(seed_position,n_seed_dimensions,activation=tf.nn.relu)
        
        seed_positions.append(seed_position)
        skew_and_var.append(tf.concat([mean_position,var_position,skew_position],axis=-1))
    
    seed_positions = tf.concat(seed_positions, axis=1)
    skew_and_var = tf.concat(skew_and_var, axis=1)
    
    edges, rot_edge = create_active_edges2(transformed_vertex_positions,seed_positions,
                                multiplier=edge_multiplicity,
                                name=this_name,
                                skew_and_var=skew_and_var,
                                fixed_frequency=0,
                                return_rot=True,
                                no_activation=True,
                                fixed_field=False)
    
    collapse_edges = gauss(edges)
    expand_edges = gauss(rot_edge)
    
    expanded_collapsed = apply_edges(propagate_features, collapse_edges, reduce_sum=True, flatten=False)
    
    expanded_collapsed = tf.transpose(expanded_collapsed,[0,1,3,2])
    
    print('applied edges       ',expanded_collapsed.shape)
        
    expanded_collapsed = half_nodes(expanded_collapsed)
    
    print('compressed features ',expanded_collapsed.shape)
    
    expanded_collapsed = tf.transpose(expanded_collapsed,[0,1,3,2])
    
    expanded_collapsed = half_nodes(expanded_collapsed)
        
    print('compressed dim      ',expanded_collapsed.shape)
    
    expanded_collapsed = tf.reshape(expanded_collapsed,[expanded_collapsed.shape[0],expanded_collapsed.shape[1],-1])
     
    expand_edges = tf.transpose(expand_edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    
    expanded_collapsed = apply_edges(expanded_collapsed, expand_edges, reduce_sum=False, flatten=False)
    
    print('applied to V       ',expanded_collapsed.shape)
    
    expanded_collapsed = tf.transpose(expanded_collapsed, perm=[0,1,4,2,3])
    
    expanded_collapsed = half_nodes(expanded_collapsed)
    
    print('compressed V F     ',expanded_collapsed.shape)
    
    expanded_collapsed = tf.reshape(expanded_collapsed,[expanded_collapsed.shape[0],
                                                        expanded_collapsed.shape[1],
                                                        expanded_collapsed.shape[2],-1])
      
    expanded_collapsed = half_nodes(expanded_collapsed)
    print('coll and comp. V   ',expanded_collapsed.shape)
        
    expanded_collapsed = tf.reshape(expanded_collapsed,[expanded_collapsed.shape[0],expanded_collapsed.shape[1],-1])
    
    expanded_collapsed = half_nodes(expanded_collapsed)
    
    print('comp. V            ',expanded_collapsed.shape)
        
    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)
    
    print('out V              ',expanded_collapsed.shape)
    
    return expanded_collapsed, seed_positions 
    
    
    

    
    
    
    
def sparse_conv_aggregator_simple(vertices_in, 
                       n_aggregators, 
                       nfilters, 
                       npropagate,
                       nspacefilters=32, 
                       nspacedim=3,
                       collapse_dropout=-1,
                       expand_dropout=-1, 
                       is_training=False,
                       compress_before_propagate=True,
                       use_edge_properties=-1,
                       return_aggregators=False,
                       input_aggregators=None,
                       weighted_aggregator_positions=False):
    
    if input_aggregators is not None:
        print('input_aggregators',input_aggregators.shape)
        assert int(input_aggregators.shape[1]) >= n_aggregators
    
    global _sparse_conv_naming_index
    '''
    '''
    #for later
    _sparse_conv_naming_index+=1
    
    
    trans_space = apply_space_transform(vertices_in, nspacefilters,nspacedim,activation=tf.nn.tanh) # Just a couple of dense layers
    
    trans_vertices = tf.layers.dense(vertices_in,npropagate,activation=tf.nn.relu) # Just dense again
    
    seed_positions =[]
    seed_properties=[]
    ### create aggregator positions...
    for i in range(n_aggregators):
        weights = vertices_in
        weights = tf.layers.dense(weights, weights.shape[-1], activation=tf.nn.relu)
        while weights.shape[-1] > 7:
            weights = half_nodes(weights)
            print('weights ',weights.shape)
        weights = tf.layers.dense(weights, 1, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(0, 0.1))+1.
        
        if weighted_aggregator_positions:
            weight_sum = tf.reduce_sum(weights,axis=1,keepdims=True)
        
            mean_position = tf.reduce_sum(trans_space * weights,axis=1, keepdims=True)/weight_sum
            diff_to_mean = trans_space-mean_position
            pos_sq = diff_to_mean **2
            var_position  = tf.reduce_sum(pos_sq * weights,axis=1, keepdims=True)/weight_sum
            skew_position  = tf.reduce_sum(pos_sq * diff_to_mean * weights,axis=1, keepdims=True)/weight_sum
            
            seed_property = tf.concat([mean_position,var_position,skew_position],axis=-1)
            print('seed_property',seed_property.shape)
            seed_position = 2.*tf.layers.dense(seed_property,nspacedim,activation=tf.nn.tanh,
                                            kernel_initializer=tf.random_normal_initializer(0, 0.1))
            
        else:
            weights = tf.reshape(weights, [weights.shape[0],weights.shape[1]])
            print('weights last',weights.shape)
            
            #input_aggregators
            seed_position = select_based_on(trans_space,    weights, n_select=1)
            seed_property = select_based_on(trans_vertices, weights, n_select=1)
        if input_aggregators is not None:
            agg = tf.expand_dims(input_aggregators[:,i,:], axis=1)
            seed_position = tf.layers.dense(tf.concat([seed_position,agg], axis=-1), nspacedim)
            seed_property = tf.layers.dense(tf.concat([seed_property,agg], axis=-1), npropagate)
        
        seed_positions.append(seed_position)
        seed_properties.append(seed_property)
        
    
    seed_trans_space = tf.concat(seed_positions, axis=1)
    seed_properties  = tf.concat(seed_properties, axis=1)
    ###

    edges = create_edges(trans_space,seed_trans_space,n_properties=use_edge_properties) # BxVxV'xF


    expanded_collapsed = apply_edges(trans_vertices, edges, reduce_sum=True, flatten=True) # [BxVxF]
    if not weighted_aggregator_positions:
        expanded_collapsed = tf.concat([expanded_collapsed,seed_properties],axis=-1)
    if collapse_dropout>0:
        expanded_collapsed = tf.layers.dropout(expanded_collapsed,rate=collapse_dropout,training=is_training)
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,nfilters, activation=tf.nn.relu)
        
    aggregators = expanded_collapsed
    print('expanded_collapsed',expanded_collapsed.shape)
    #propagate back, transposing the edges does the trick, now they point from Nseeds to Nvertices
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    expanded_collapsed = apply_edges(expanded_collapsed, edges, reduce_sum=False, flatten=True)
    if expand_dropout>0:
        expanded_collapsed = tf.layers.dropout(expanded_collapsed,rate=expand_dropout,training=is_training)
    if compress_before_propagate:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,nfilters, activation=tf.nn.relu)
    

    #combien old features with new ones
    feature_layerout = tf.concat([vertices_in,trans_space,expanded_collapsed],axis=-1)
    feature_layerout = tf.layers.dense(feature_layerout,nfilters,activation=tf.nn.relu,
                                       kernel_initializer=NoisyEyeInitializer)
    
    if return_aggregators:
        return feature_layerout, aggregators
    
    return feature_layerout
    
    
    

def sparse_conv_make_neighbors_simple(vertices_in, 
                                      num_neighbors=16, 
                                      n_filters=[16], 
                                      n_output=16,
                                      edge_filters=[64],
                                      space_transformations=[32],
                                      train_global_space=False,
                                      individual_filters=False,
                                      ):
    
    assert len(n_filters)
    global _sparse_conv_naming_index
    #for later
    _sparse_conv_naming_index+=1
    
    trans_space = vertices_in
    if train_global_space:
        trans_space = trans_space[0:1,:,:]
        trans_space = tf.concat([trans_space[:,:,0:3],trans_space[:,:,4:]],axis=-1) #CHECK
    
    for i in range(len(space_transformations)):
        f = space_transformations[i]
        space_activation = tf.nn.relu
        if i == len(space_transformations) - 1:
            space_activation = None
        trans_space = tf.layers.dense(trans_space,f,activation=space_activation)
    
    print('trans_space',trans_space.shape)
    
    if train_global_space:
        indexing, _ = indexing_tensor_2(trans_space, num_neighbors, n_batch=vertices_in.shape[0])
        trans_space = tf.tile(trans_space,[vertices_in.shape[0],1,1])
    else:
        indexing, _ = indexing_tensor_2(trans_space, num_neighbors)


    neighbour_space = tf.gather_nd(trans_space, indexing)
    
    #build edges manually
    expanded_trans_space = tf.expand_dims(trans_space, axis=2)
    diff = expanded_trans_space - neighbour_space
    edges = diff #add_rot_symmetric_distance(diff)
    
    edges = gauss_of_lin(edges*edges)
    #maybe not needed with tensordot
    # BxVxNxD
    
    #edges = tf.expand_dims(edges,axis=3)
    
    
    #transform the edges a few times a la edgeconv and then use each output node as multiplier
    #use tensordot for memory
    
    trans_vertices = vertices_in
    for f in n_filters:
        trans_vertices = tf.layers.dense(trans_vertices,f,activation=tf.nn.relu)
        #some dense on the vertex input
    
    neighbour_vertices = tf.gather_nd(trans_vertices, indexing)
    for f in edge_filters:
        edges = tf.layers.dense(edges,f,activation=tf.nn.relu)
    print('edges',edges.shape)
    print('neighbour_vertices',neighbour_vertices.shape)
    
    #updated_vertices = tf.einsum('ijkl,abkd->ijld', edges, neighbour_vertices)/float(num_neighbors)
    edges = tf.expand_dims(edges, axis=4)
    neighbour_vertices = tf.expand_dims(neighbour_vertices, axis=3)
    print('edges',edges.shape)
    print('neighbour_vertices',neighbour_vertices.shape)

    updated_vertices = tf.reshape(neighbour_vertices*edges,[neighbour_vertices.shape[0],neighbour_vertices.shape[1],-1])

    updated_vertices = tf.concat([trans_space,updated_vertices],axis=-1)
    print('updated_vertices',updated_vertices.shape)
    return tf.layers.dense(updated_vertices,n_output,activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)

    
    

def sparse_conv_make_neighbors_simple_multipass(vertices_in, 
                                      num_neighbors=16, 
                                      n_propagate=8,
                                      n_filters=[16], 
                                      n_output=16,
                                      edge_filters=[64],
                                      space_transformations=[32],
                                      train_global_space=False,
                                      ):
    
    assert len(n_filters)
    global _sparse_conv_naming_index
    #for later
    _sparse_conv_naming_index+=1
    
    trans_space = vertices_in
    if train_global_space:
        trans_space = trans_space[0:1,:,:]
        trans_space = tf.concat([trans_space[:,:,0:3],trans_space[:,:,4:]],axis=-1) #CHECK
    
    for i in range(len(space_transformations)):
        f = space_transformations[i]
        space_activation = tf.nn.relu
        if i == len(space_transformations) - 1:
            space_activation = None
        trans_space = tf.layers.dense(trans_space,f,activation=space_activation)
    
    print('trans_space',trans_space.shape)
    
    if train_global_space:
        indexing, _ = indexing_tensor_2(trans_space, num_neighbors, n_batch=vertices_in.shape[0])
        trans_space = tf.tile(trans_space,[vertices_in.shape[0],1,1])
    else:
        indexing, _ = indexing_tensor_2(trans_space, num_neighbors)


    neighbour_space = tf.gather_nd(trans_space, indexing)
    
    #build edges manually
    expanded_trans_space = tf.expand_dims(trans_space, axis=2)
    diff = expanded_trans_space - neighbour_space
    edges = diff #add_rot_symmetric_distance(diff)
    
    edges = gauss_of_lin(edges*edges)
    
    updated_vertices = tf.layers.dense(vertices_in,n_propagate)
    edges_orig=edges
    for i in range(len(n_filters)):
        trans_vertices = updated_vertices #tf.layers.dense(updated_vertices,n_filters[i],activation=tf.nn.relu)
        neighbour_vertices = tf.gather_nd(trans_vertices, indexing)
        edges = tf.layers.dense(edges_orig,edge_filters[i],activation=tf.nn.relu)
        edges = tf.expand_dims(edges, axis=4)
        neighbour_vertices = tf.expand_dims(neighbour_vertices, axis=3)
        vertex_update = tf.reshape(neighbour_vertices*edges,[neighbour_vertices.shape[0],neighbour_vertices.shape[1],-1])
        updated_vertices = tf.concat([updated_vertices,vertex_update], axis=-1)
        updated_vertices = tf.layers.dense(updated_vertices,n_filters[i],activation=tf.nn.relu)
        
    #some dense on the vertex input
    
    updated_vertices = tf.concat([trans_space,updated_vertices],axis=-1)
    print('updated_vertices',updated_vertices.shape)
    return tf.layers.dense(updated_vertices,n_output,activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)

    
    
    
    
def sparse_conv_hidden_aggregators(vertices_in,
                                   n_aggregators,
                                   n_filters,
                                   pre_filters=[],
                                   n_propagate=-1,
                                   plus_mean=False,
                                   return_agg=False
                                   ):   
    vertices_in_orig = vertices_in
    trans_vertices = vertices_in
    for f in pre_filters:
        trans_vertices = tf.layers.dense(trans_vertices,f,activation=tf.nn.relu)
    
    if n_propagate>0:
        vertices_in = tf.layers.dense(vertices_in,n_propagate,activation=None)
    
    agg_nodes = tf.layers.dense(trans_vertices,n_aggregators,activation=None) #BxVxNA, vertices_in: BxVxF
    agg_nodes = gauss(agg_nodes)
    #agg_nodes=sprint(agg_nodes,"agg_nodes")
    vertices_in = tf.concat([vertices_in,agg_nodes], axis=-1)
    
    edges = tf.expand_dims(agg_nodes,axis=3) # BxVxNAx1
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    
    print('edges',edges.shape)
    print('vertices_in',vertices_in.shape)
    
    vertices_in_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True)#,aggregation_function=tf.reduce_mean)# [BxNAxF]
    vertices_in_mean_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True ,aggregation_function=tf.reduce_mean)# [BxNAxF]
    
    #vertices_in_collapsed = sprint(vertices_in_collapsed,'vertices_in_collapsed')
    if plus_mean:
        vertices_in_collapsed= tf.concat([vertices_in_collapsed,vertices_in_mean_collapsed],axis=-1 )
    print('vertices_in_collapsed',vertices_in_collapsed.shape)

    if return_agg:
        return vertices_in_collapsed
    
    edges = tf.transpose(edges, perm=[0,2, 1,3]) # [BxVxV'xF]
    
    expanded_collapsed = apply_edges(vertices_in_collapsed, edges, reduce_sum=False, flatten=True)# [BxVxF]
    
    print('expanded_collapsed',expanded_collapsed.shape)
    #expanded_collapsed = sprint(expanded_collapsed,'expanded_collapsed')
    expanded_collapsed = tf.concat([vertices_in_orig,expanded_collapsed,agg_nodes], axis=-1)
    
    print('expanded_collapsed2',expanded_collapsed.shape)
    
    #merged_out = tf.layers.dense(expanded_collapsed,n_filters,activation=tf.nn.tanh)
    merged_out = high_dim_dense(expanded_collapsed,n_filters,activation=tf.nn.tanh)
    
    return merged_out, agg_nodes
    
    
def sparse_conv_multi_neighbours(vertices_in,
                                   n_neighbours,
                                   n_dimensions,
                                   n_filters,
                                   n_propagate=-1,
                                   individual_conv=False,
                                   total_distance=False,
                                   plus_mean=False):
    
    
    trans_vertices = vertices_in
    
    if n_propagate>0:
        vertices_prop = high_dim_dense(trans_vertices,n_propagate,activation=None)
        
    neighb_dimensions = high_dim_dense(trans_vertices,n_dimensions,activation=None) #BxVxND, 
    #neighb_dimensions=sprint(neighb_dimensions,'neighb_dimensions')
    
    def collapse_to_vertex(indexing,distance,vertices,indiv_conv):
        neighbours = tf.gather_nd(vertices, indexing)  #BxVxNxF
        distance = tf.expand_dims(distance,axis=3)
        if not total_distance:
            distance = distance*1e5
        else:
            distance = distance*10.
        #distance = sprint(distance,'distance')
        #don't take the origin vertex - will be mixed later
        edges = gauss_of_lin(distance)[:,:,1:,:]
        #edges = sprint(edges,'edges')
        neighbours = neighbours[:,:,1:,:]
        scaled_feat = edges*neighbours
        collapsed = tf.reduce_max(scaled_feat, axis=2)
        collapsed_mean = tf.reduce_mean(scaled_feat,axis=2)
        if plus_mean:
            collapsed = tf.concat([collapsed,collapsed_mean],axis=-1)
        if indiv_conv:
            collapsed = tf.concat([collapsed, tf.reshape(neighbours,[neighbours.shape[0],neighbours.shape[1],-1])],axis=-1)
        return collapsed
    
    out_per_dim = []
    if total_distance:
        indexing, distance = indexing_tensor_2(neighb_dimensions, n_neighbours)
        out_per_dim.append(collapse_to_vertex(indexing,distance,vertices_prop,individual_conv))
    else:
        neighb_dimensions_exp = tf.expand_dims(neighb_dimensions,axis=3) #BxVxNDx1
        for d in range(n_dimensions):
            indexing, distance = indexing_tensor_2(neighb_dimensions_exp[:,:,d,:], n_neighbours)
            out_per_dim.append(collapse_to_vertex(indexing,distance,vertices_prop,individual_conv))
        #make it a weighted mean mean of weighted
        #add a conv part where IN ADDITION to the weighted mean, the features are ordered by distance (just use neighbouts as they are)
    
    collapsed = tf.concat(out_per_dim,axis=-1)
    updated_vertices = tf.concat([vertices_in,collapsed],axis=-1)
    print('updated_vertices',updated_vertices.shape)
    return high_dim_dense(updated_vertices,n_filters,activation=tf.nn.tanh), neighb_dimensions
    
    #
    # use a similar reduction to one value to determine neighbour relations 
    # (but do this for multiple dimensions - say neighbours in x, neighbours in y, .. etc. not in x^2+y^2
    # that should be as memory inefficient but might bring some perf
    #
    

def sparse_conv_multi_neighbours_reference(vertices_in,
                                   n_neighbours,
                                   n_dimensions,
                                   n_filters,
                                   n_propagate=-1):
    
    return sparse_conv_multi_neighbours(vertices_in,
                                   n_neighbours,
                                   n_dimensions,
                                   n_filters,
                                   n_propagate=n_propagate,
                                   individual_conv=False,
                                   total_distance=True,
                                   plus_mean=True)



def sparse_conv_hidden_aggregators_reference(vertices_in,
                                   n_aggregators,
                                   n_filters,
                                   n_propagate=-1
                                   ):
    return sparse_conv_hidden_aggregators(vertices_in,
                                   n_aggregators,
                                   n_filters,
                                   pre_filters=[],
                                   n_propagate=n_propagate,
                                   plus_mean=True
                                   )


    
    

import readers.indices_calculated as ic    
def construct_binning16(vertices_in):
    
    batch_size = int(vertices_in.shape[0])
    max_entries = int(vertices_in.shape[1])
    nfeat = int(vertices_in.shape[2])
    
    
    batch_indices = np.arange(batch_size)
    batch_indices = np.tile(batch_indices[..., np.newaxis], reps=(1, max_entries))[..., np.newaxis]

    indexing_array = \
        np.concatenate((ic.x_bins_beta_calo_16[:, np.newaxis], ic.y_bins_beta_calo_16[:, np.newaxis], ic.l_bins_beta_calo_16[:, np.newaxis],
                        ic.d_indices_beta_calo_16[:, np.newaxis]),
                       axis=1)[np.newaxis, ...]

    indexing_array = np.tile(indexing_array, reps=[batch_size, 1, 1])
    indexing_array = np.concatenate((batch_indices, indexing_array), axis=2).astype(np.int64)


    indexing_array2 = \
        np.concatenate((ic.x_bins_beta_calo[:, np.newaxis], ic.y_bins_beta_calo[:, np.newaxis], ic.l_bins_beta_calo[:, np.newaxis],
                        ic.d_indices_beta_calo[:, np.newaxis]),
                       axis=1)[np.newaxis, ...]

    indexing_array2 = np.tile(indexing_array2, reps=[batch_size, 1, 1])
    indexing_array2 = np.concatenate((batch_indices, indexing_array2), axis=2).astype(np.int64)

    result = tf.scatter_nd(indexing_array, vertices_in, shape=(batch_size, 16, 16, 20, 4, nfeat ))
    result = tf.reshape(result, [batch_size, 16, 16, 20, -1])

    return result, indexing_array

def construct_binning20(vertices_in):
    
    batch_size = int(vertices_in.shape[0])
    max_entries = int(vertices_in.shape[1])
    nfeat = int(vertices_in.shape[2])
    
    
    batch_indices = np.arange(batch_size)
    batch_indices = np.tile(batch_indices[..., np.newaxis], reps=(1, max_entries))[..., np.newaxis]

    indexing_array = \
        np.concatenate((ic.x_bins_beta_calo_20[:, np.newaxis], ic.y_bins_beta_calo_20[:, np.newaxis], ic.l_bins_beta_calo_20[:, np.newaxis],
                        ic.d_indices_beta_calo_20[:, np.newaxis]),
                       axis=1)[np.newaxis, ...]

    indexing_array = np.tile(indexing_array, reps=[batch_size, 1, 1])
    indexing_array = np.concatenate((batch_indices, indexing_array), axis=2).astype(np.int64)


    indexing_array2 = \
        np.concatenate((ic.x_bins_beta_calo[:, np.newaxis], ic.y_bins_beta_calo[:, np.newaxis], ic.l_bins_beta_calo[:, np.newaxis],
                        ic.d_indices_beta_calo[:, np.newaxis]),
                       axis=1)[np.newaxis, ...]

    indexing_array2 = np.tile(indexing_array2, reps=[batch_size, 1, 1])
    indexing_array2 = np.concatenate((batch_indices, indexing_array2), axis=2).astype(np.int64)

    result = tf.scatter_nd(indexing_array, vertices_in, shape=(batch_size, 20, 20, 20, 1, nfeat ))
    result = tf.reshape(result, [batch_size, 20, 20, 20, -1])

    return result, indexing_array
    
def sparse_conv_global_exchange_binned(binned_in):
    in_shape = binned_in.shape.as_list()
    reshaped = tf.reshape(binned_in, [in_shape[0], -1, in_shape[-1]])
    reshaped = sparse_conv_global_exchange(reshaped)
    out =  tf.reshape(reshaped,in_shape[:-1]+[2*in_shape[-1]])
    return out

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
