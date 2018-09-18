import tensorflow as tf
from ops.sparse_conv import *
from ops.neighbors import *


def printAndEval(prtext, x, sess):
    
    print(prtext)
    result = sess.run(x)
    print(result)

#BxEx3
test_spatial_global = tf.constant([[[1, 2, 3],
                                    [1.1, 2.1, 3.1],
                                    [1, 1.9, 3],
                                    [0.1, 3, 2],
                                    [0, 4, 2]],
                                    ##this one is the 'zeroed out' one
                                    [[0.1, 12, 13],
                                     [0.3, 12.1, 13.1],
                                     [0, 11.9, 13],
                                     [0, 13, 12],
                                     [0, 14, 12]]],dtype=tf.float32)

test_spatial_local = tf.constant([[[10, 30],
                                   [20, 50],
                                   [10, 60],
                                   [20, 10],
                                   [00, 20]],
                                   [[110, 130],
                                    [120, 150],
                                    [110, 160],
                                    [120, 110],
                                    [100, 120]]],dtype=tf.float32)

#BxExF
test_features = tf.constant([[[.1, .2],
                              [.2, .3],
                              [.1, .4],
                              [.2, .2],
                              [.0, .4]],
                              [[1.1, 1.2],
                               [1.2, 1.3],
                               [1.1, 1.4],
                               [1.2, 1.2],
                               [1.0, 1.4]]],dtype=tf.float32)


distancetozero  =  tf.constant([[[.1],
                              [.2],
                              [.1],
                              [.2],
                              [.0]],
                              [[1.1],
                               [1.2],
                               [1.1],
                               [1.2],
                               [1.0]]],dtype=tf.float32)


n_batch=2
n_max_entries=5
output_all=8
num_entries=5

net = construct_sparse_io_dict(test_features, test_spatial_global, test_spatial_local, num_entries)

#assumes euclidean
print('input shape spat features: ',test_spatial_global.shape)

with tf.Session() as sess:
    
    matrix = nearest_neighbor_matrix(test_spatial_global,3)
    result = sess.run(matrix)
    print(result)
    
    indexing = indexing_tensor(test_spatial_global,4)
    result = sess.run(indexing)
    print(result)
    
    
    indexing = indexing[:,:,1]
    result = sess.run(indexing)
    print(result)
    
    
    gathered_feat=tf.gather_nd(test_features, indexing)
    printAndEval('gathered_feat',gathered_feat,sess)
    
    exit()
    
    indices = sort_last_dim_tensor(distancetozero)
    printAndEval('indices',indices,sess)
    
    gathered_feat=tf.gather_nd(test_features, indices)
    printAndEval('gathered_feat',gathered_feat,sess)
    
    sorting_in_x = sort_last_dim_tensor(test_features[:,:,0])
    printAndEval('sorting_in_x ',sorting_in_x,sess)
    sorted= tf.gather_nd(test_features,sorting_in_x)
    printAndEval('sorted ',sorted,sess)
    
    exit()
    zerostest = tf.equal(test_spatial_global[:,:,0], 0)
    
    zerostest = tf.reshape(zerostest, [n_batch*num_entries])
    test_features = tf.reshape(test_features, [n_batch*num_entries,  2])
    
    
    exit()
    printAndEval('zerostest ',zerostest,sess)
    printAndEval('test_features ',test_features,sess)
    
    sc = tf.cond(zerostest, lambda: tf.zeros_like(test_features), lambda: test_features)
    printAndEval('sc ',sc,sess)
    #sc = sparse_conv_2(net, num_neighbors=3, num_filters=2, n_prespace_conditions=2)
    
    
    
    
    