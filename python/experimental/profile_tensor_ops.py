import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.profiler import option_builder
import argparse

parser = argparse.ArgumentParser(description='Profile tf ops')
parser.add_argument('mode',
                    help="Modes: 1. Dense 2. Custom matmul dense 3. 1D conv 4. Multiplications")
args = parser.parse_args()



batch = 100
input_features = 2048
mid_features = 2048
num_iterations = 10
profiler_output_file = '/afs/cern.ch/work/s/sqasim/workspace_4/ProfilingResults/alpha/alpha'


weight_values = {
    'wc1': tf.Variable(tf.random_normal([input_features, mid_features])),
    'wc2': tf.Variable(tf.random_normal([mid_features, 1])),
}

graph_dense = tf.random_normal(shape=(batch, input_features))

if int(args.mode) == 1:
    graph_dense = tf.layers.dense(graph_dense, units=mid_features, use_bias=False, activation=tf.nn.relu)
    graph_dense = tf.layers.dense(graph_dense, units=1, use_bias=False)
elif int(args.mode) == 2:
    graph_dense = tf.reduce_sum(graph_dense[:, :, tf.newaxis] * weight_values['wc1'][tf.newaxis, :, :], axis=1)
    graph_dense = tf.reduce_sum(graph_dense[:, :, tf.newaxis] * weight_values['wc2'][tf.newaxis, :, :], axis=1)
elif int(args.mode) == 3:
    graph_dense = tf.layers.conv1d(graph_dense[:,tf.newaxis, :], mid_features, kernel_size=1)
    graph_dense = tf.layers.conv1d(graph_dense, 1, kernel_size=1)
    graph_dense = graph_dense[:, 0, :]
elif int(args.mode) == 4:
    graph_dense = tf.reduce_prod(graph_dense[:, :, tf.newaxis] * weight_values['wc1'][tf.newaxis, :, :], axis=1)
    graph_dense = tf.reduce_prod(graph_dense[:, :, tf.newaxis] * weight_values['wc2'][tf.newaxis, :, :], axis=1)
else:
    0/0


graph_dense = tf.losses.mean_squared_error(tf.random_normal(shape=(batch, 1)), graph_dense)


graph_dense = tf.train.AdamOptimizer().minimize(graph_dense)

init=tf.global_variables_initializer()

print("Graphs initialized")


with tf.Session() as sess:
    sess.run(init)
    profiler = Profiler(sess.graph)

    for iteration_number in range(num_iterations):
        print("Iteration ", iteration_number)
        run_meta = tf.RunMetadata()

        sess.run(graph_dense,
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



