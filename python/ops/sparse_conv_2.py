import tensorflow as tf
from .neighbors import euclidean_squared,indexing_tensor, indexing_tensor_2, sort_last_dim_tensor, get_sorted_vertices_ids
from ops.nn import *
import numpy as np
from .initializers import NoisyEyeInitializer
from .activations import gauss_of_lin, gauss_times_linear, sinc, open_tanh, asymm_falling
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
    


def sprint(tensor,pstr):
    return tensor
    return tf.Print(tensor,[tensor],pstr,summarize=300)

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
        scaled_colours_in = tf.log(colours_in+1)/10.
    
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
    
def sparse_conv_multipl_dense(in_tensor, nfilters, activation=None, kernel_initializer=None, name=None):
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
    weights = tf.get_variable(name+"_weights", [nfilters*int(in_tensor.shape[-1])],dtype=tf.float32,
                                        initializer=kernel_initializer)/10.
    weights = tf.reshape(weights, weights_shape)
    
    print('weights',weights.shape)
    
    bias = tf.get_variable(name+"_bias", [nfilters],dtype=tf.float32,
                                        initializer=tf.zeros_initializer)    
    
    expanded_in_tensor = tf.expand_dims(in_tensor, axis=-2)

    weights = tf.tile(weights, batch_etc+[1,1])
    
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
    
    
def apply_edges(vertices, edges, reduce_sum=True, flatten=True): 
    '''
    edges are naturally BxVxV'xF
    vertices are BxVxF'  or BxV'xF'
    This function returns BxVxF'' if flattened and summed
    '''
    edges = tf.expand_dims(edges,axis=3)
    vertices = tf.expand_dims(vertices,axis=1)
    vertices = tf.expand_dims(vertices,axis=4)

    out = edges*vertices # [BxVxV'x1xF] x [Bx1xV'xF'x1] = [BxVxV'xFxF']

    if reduce_sum:
        out = tf.reduce_sum(out,axis=2)/float(int(out.shape[2]))
    if flatten:
        out = tf.reshape(out,shape=[out.shape[0],out.shape[1],-1])
    
    return out

 
def apply_space_transform(vertices, units_transform, output_dimensions,
                          depth=1): 
    trans_space = vertices
    for i in range(depth):
        trans_space = tf.layers.dense(trans_space/10.,units_transform,activation=open_tanh,
                                       kernel_initializer=NoisyEyeInitializer)
        trans_space = trans_space*10.
    trans_space = tf.layers.dense(trans_space*10.,output_dimensions,activation=None,
                                  kernel_initializer=NoisyEyeInitializer, use_bias=False)
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
                               train_space=True
                               ):
    
    assert merge_neighbours <= num_neighbors
    global _sparse_conv_naming_index
    space_initializer = None #NoisyEyeInitializer
    
    #for later
    _sparse_conv_naming_index+=1
    
    space_transformations = make_sequence(space_transformations)
    output_all = make_sequence(output_all)
    
    
    trans_space = vertices_in
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
    if train_space==False and space_transformations[-1] <= 3:
        indexing, _ = indexing_tensor_2(trans_space[0:1,:,:], num_neighbors, n_batch=trans_space.shape[0])
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
                      share_keyword=None #TBI
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
    
    vertex_out = aggregation_function(edge,axis=2)
    
    return vertex_out
    


def max_pool_on_last_dimensions(vertices_in, skip_first_features, n_output_vertices):
    
    all_features = vertices_in[:,:,skip_first_features:-1]
    
    _, I = tf.nn.top_k(tf.reduce_max(all_features, axis=2), n_output_vertices)
    I = tf.expand_dims(I, axis=2)

    batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, vertices_in.shape[0]), axis=1), axis=1)
    batch_range = tf.tile(batch_range, [1,n_output_vertices, 1])
    _indexing_tensor = tf.concat([batch_range, I], axis=2)
    
    return tf.gather_nd(vertices_in, _indexing_tensor)
    

def create_active_edges2(vertices_a, vertices_b, name,multiplier=1,skew_and_var=None,fixed_frequency=-1,
                         no_activation=False, return_rot=False):
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
                             seed_filters=[],
                             out_filters=[],
                             weight_filters=[],
                             edge_multiplicity=1):
    
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
        
        max_position = select_based_on(transformed_vertex_positions, tf.reduce_max(weights,axis=1), n_select=1)
        mean_position = tf.reduce_sum(transformed_vertex_positions * weights,axis=1, keepdims=True)/weight_sum
        diff_to_mean = transformed_vertex_positions-mean_position
        pos_sq = diff_to_mean **2
        var_position  = tf.reduce_sum(pos_sq * weights,axis=1, keepdims=True)/weight_sum
        skew_position  = tf.reduce_sum(pos_sq * diff_to_mean * weights,axis=1, keepdims=True)/weight_sum
        
        seed_position = tf.concat([max_position,mean_position,var_position,skew_position],axis=-1)
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
            
    expanded_collapsed = tf.concat([vertices_in,expanded_collapsed],axis=-1)

    for f in out_filters:
        expanded_collapsed = tf.layers.dense(expanded_collapsed,f,activation=tf.nn.relu,kernel_initializer=NoisyEyeInitializer)
        
    expanded_collapsed = tf.layers.dense(expanded_collapsed,n_filters, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)
    
    print('sparse_conv_moving_seeds4 out',expanded_collapsed.shape)
    return expanded_collapsed, seed_positions

    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
