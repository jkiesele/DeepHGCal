import tensorflow as tf
from .neighbors import euclidean_squared,indexing_tensor, indexing_tensor_2, sort_last_dim_tensor, get_sorted_vertices_ids
from ops.nn import *
import numpy as np
from ops.sparse_conv import NoisyEyeInitializer
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

def sparse_conv_normalise(sparse_dict):
    colours_in, space_global, space_local, num_entries = sparse_dict['all_features'], \
                                                         sparse_dict['spatial_features_global'], \
                                                         sparse_dict['spatial_features_local'], \
                                                         sparse_dict['num_entries']
    
    scaled_colours_in = colours_in*1e-4
    print('space_global',space_global.shape)
    print('colours_in',colours_in.shape)
    print('space_local',space_local.shape)
    
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
    
    
    

def normalise_distance_matrix(AdMat):
    #return 1-tf.nn.softsign(4*AdMat)
    mat=tf.exp(-1*(tf.abs(AdMat)))
    #mat=tf.nn.softmax(-tf.abs(AdMat))
    #mat=sprint(mat, "pstr")
    return mat

def get_distance_weight(x):
    #return x
    #return tf.exp(-1*(tf.abs(x)))
    return tf.exp(-6.*tf.sqrt(tf.abs(x)+1e-4))*x*3.*36 #integral 0->inf = 1
    
    

def create_edges(vertices_a, vertices_b, add_radial=True):
    #BxVxF
    expanded_vertices_a = tf.expand_dims(vertices_a, axis=1)
    expanded_vertices_b = tf.expand_dims(vertices_b, axis=2)
    raw_difference = expanded_vertices_a - expanded_vertices_b
    #calculate explicitly rotational symmetric on
    
    rot_raw_difference = tf.reduce_sum(raw_difference*raw_difference,axis=-1)
    rot_raw_difference = tf.expand_dims(rot_raw_difference,axis=3)
    
    edges = tf.concat([rot_raw_difference,raw_difference],axis=-1)
    return get_distance_weight(edges)
    
    
def apply_edges(vertices, edges, reduce_sum=True, flatten=True): 
    '''
    edges are naturally BxVxV'xF
    vertices are BxVxF'  or BxV'xF'
    This function returns BxVxF'' if flattened and summed
    '''
    edges = tf.expand_dims(edges,axis=3)
    vertices = tf.expand_dims(vertices,axis=1)
    vertices = tf.expand_dims(vertices,axis=4)
    
    out = edges*vertices
    if reduce_sum:
        out = tf.reduce_sum(out,axis=2)/float(int(out.shape[2]))
    if flatten:
        out = tf.reshape(out,shape=[out.shape[0],out.shape[1],-1])
    
    return out

 
def apply_space_transform(vertices, units_transform, output_dimensions): 
    
    trans_space = tf.layers.dense(vertices/10.,units_transform,activation=tf.nn.tanh,
                                   kernel_initializer=NoisyEyeInitializer)
    trans_space = tf.layers.dense(trans_space*10.,output_dimensions,activation=None,
                                  kernel_initializer=NoisyEyeInitializer, use_bias=False)
    return trans_space
########

def sparse_conv_seeded3(vertices_in, 
                       seed_indices, 
                       nfilters, 
                       nspacefilters=1, 
                       nspacedim=3, 
                       seed_talk=True):
    global _sparse_conv_naming_index
    '''
    '''
    #for later
    _sparse_conv_naming_index+=1
    
    seedselector = make_batch_selection(seed_indices)
    
    trans_space = apply_space_transform(vertices_in, nspacefilters,nspacedim)
    
    seed_trans_space = tf.gather_nd(trans_space,seedselector)
    
    edges = create_edges(trans_space,seed_trans_space)
    
    expanded_collapsed = apply_edges(vertices_in, edges, reduce_sum=True, flatten=True)
   
    #add back seed features
    seed_all_features = tf.gather_nd(vertices_in,seedselector)
    
    #simple dense check this part
    #maybe add additional dense
    if seed_talk:
        #seed space transform?
        seed_edges = create_edges(seed_trans_space,seed_trans_space)
        trans_seeds = apply_edges(seed_all_features, seed_edges, reduce_sum=True, flatten=True)
        seed_merged_features = tf.concat([seed_all_features,trans_seeds],axis=-1)
        seed_all_features = tf.layers.dense(seed_merged_features,seed_all_features.shape[2],activation=tf.nn.relu,
                                            kernel_initializer=NoisyEyeInitializer)
    
    #compress
    
    expanded_collapsed = tf.concat([seed_all_features,expanded_collapsed],axis=-1)
    expanded_collapsed = tf.layers.dense(expanded_collapsed,nfilters, activation=tf.nn.tanh)
    
    #propagate back, transposing the edges does the trick, now they point from Nseeds to Nvertices
    edges = tf.transpose(edges, perm=[0,2, 1,3])
    expanded_collapsed = apply_edges(expanded_collapsed, edges, reduce_sum=False, flatten=True)
    

    #combien old features with new ones
    feature_layerout = tf.concat([vertices_in,expanded_collapsed,],axis=-1)
    feature_layerout = tf.layers.dense(feature_layerout/10.,nfilters, activation=tf.nn.tanh,kernel_initializer=NoisyEyeInitializer)
    feature_layerout = feature_layerout*10.
    feature_layerout=sprint(feature_layerout, 'feature_layerout')
    return feature_layerout




def sparse_conv_make_neighbors2(vertices_in, num_neighbors=10, 
                               output_all=15, space_transformations=10):
    global _sparse_conv_naming_index
    #for later
    _sparse_conv_naming_index+=1
    
    space_transformations = make_sequence(space_transformations)
    output_all = make_sequence(output_all)
    
    trans_space = vertices_in
    for i in range(len(space_transformations)):
        if i< len(space_transformations)-1:
            trans_space = tf.layers.dense(trans_space/10.,space_transformations[i],activation=tf.nn.tanh,
                                       kernel_initializer=NoisyEyeInitializer)
            trans_space*=10.
        else:
            trans_space = tf.layers.dense(trans_space,space_transformations[i],activation=None,
                                       kernel_initializer=NoisyEyeInitializer)

    indexing, distance = indexing_tensor_2(trans_space, num_neighbors)
    
    neighbour_space = tf.gather_nd(trans_space, indexing)
    print('neighbour_space',neighbour_space.shape)
    edges = create_edges(trans_space, neighbour_space)
        
    updated_vertices = vertices_in
    for f in output_all:
        vertex_with_neighbours = tf.gather_nd(updated_vertices, indexing)
        flattened_gathered = apply_edges(vertex_with_neighbours, edges, reduce_sum=True, flatten=True)
        updated_vertices = tf.layers.dense(flattened_gathered, f, activation=tf.nn.relu) 


    return 


    
    