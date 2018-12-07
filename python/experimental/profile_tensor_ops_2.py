import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.profiler import option_builder
import argparse
import numpy as np


def euclidean_squared(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * tf.matmul(A, tf.transpose(B, perm=[0, 2, 1]))  # -2ab term
    dotA = tf.expand_dims(tf.reduce_sum(A * A, axis=2), axis=2)  # a^2 term
    dotB = tf.expand_dims(tf.reduce_sum(B * B, axis=2), axis=1)  # b^2 term
    return tf.abs(sub_factor + dotA + dotB)


def euclidean_squared_dumb(A, B):
    """
    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Returns:
    A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
    jth in second set.

    """

    shape_A = A.get_shape().as_list()
    shape_B = B.get_shape().as_list()

    assert (A.dtype == tf.float32 or A.dtype == tf.float64) and (B.dtype == tf.float32 or B.dtype == tf.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]  # and shape_A[1] == shape_B[1]

    B_trans = tf.transpose(tf.expand_dims(B,axis=3), perm=[0, 3, 1, 2])
    A_exp = tf.expand_dims(A,axis=2)
    diff = A_exp-B_trans
    distance = tf.reduce_sum(tf.square(diff),axis=-1)
    #to avoid rounding problems and keep it strict positive
    distance= tf.abs(distance)
    return distance


batch = 100
vertices = 2679
num_features = 16
num_iterations = 10
profiler_output_file = '/afs/cern.ch/work/s/sqasim/workspace_4/ProfilingResults/alpha/alpha'


input = tf.random_normal(shape=(batch, vertices, num_features), stddev=100000)
input2 = tf.random_normal(shape=(batch, vertices, num_features), stddev=100000)
input3 = tf.random_normal(shape=(batch, vertices, num_features), stddev=100000)
graph = tf.reduce_sum(euclidean_squared(input, input), axis=-1)
graph = graph + tf.reduce_sum(euclidean_squared(input2, input), axis=-1)
graph = graph + tf.reduce_sum(euclidean_squared(input3, input), axis=-1)



init=tf.global_variables_initializer()

print("Graphs initialized")


with tf.Session() as sess:
    sess.run(init)
    profiler = Profiler(sess.graph)

    for iteration_number in range(num_iterations):
        print("Iteration ", iteration_number)
        run_meta = tf.RunMetadata()

        output = sess.run([graph],
                 options=tf.RunOptions(
                     trace_level=tf.RunOptions.FULL_TRACE),
                 run_metadata=run_meta)


        profiler.add_step(iteration_number, run_meta)

        # Profile the parameters of your model.
        profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder
                                             .trainable_variables_parameter()))

        # Or profile the timing of your model operations.
        opts = option_builder.ProfileOptionBuilder.time_and_memory()
        profiler.profile_operations(options=opts)

        # Or you can generate a timeline:
        opts = (option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(iteration_number)
                .with_timeline_output(profiler_output_file).build())
        profiler.profile_graph(options=opts)



