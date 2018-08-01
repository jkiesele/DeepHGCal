import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import matplotlib.cm as cmx
import tensorflow as tf


def get_input_feeds(file, num_max_entries=3000, num_spatial_features=3, num_spatial_local=2, num_all_features=4, num_classes=6):
    def _parse_function(example_proto):
        keys_to_features = {
            'spatial_features': tf.FixedLenFeature((num_max_entries, num_spatial_features), tf.float32),
            'spatial_local_features': tf.FixedLenFeature((num_max_entries, num_spatial_local),
                                                         tf.float32),
            'all_features': tf.FixedLenFeature((num_max_entries, num_all_features), tf.float32),
            'labels_one_hot': tf.FixedLenFeature((num_classes,), tf.int64),
            'num_entries': tf.FixedLenFeature(1, tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['spatial_features'], parsed_features['spatial_local_features'], parsed_features[
            'all_features'], parsed_features['labels_one_hot'], parsed_features['num_entries']

    file_paths = [file]
    dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
    dataset = dataset.map(_parse_function)
    iterator = dataset.make_one_shot_iterator()
    inputs = iterator.get_next()

    return inputs


def find_type(A):
    assert np.size(A) == 6

    if A[0] == 1:
        return "Electron"
    elif A[1] == 1:
        return "Muon"
    elif A[2] == 1:
        return "Charged Pion"
    elif A[3] == 1:
        return "Neutral Pion"
    elif A[4] == 1:
        return "K0 Long"
    elif A[5] == 1:
        return "K0 Short"


parser = argparse.ArgumentParser(description='Plot denoising output')
parser.add_argument('input',
                    help="Path to the file which you want to plot")
args = parser.parse_args()

file = args.input
location = 'B4'
branches = ['isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short', 'rechit_energy',
            'rechit_x', 'rechit_y', 'rechit_z']
types = ['int32', 'int32', 'int32', 'int32', 'int32', 'int32', 'float64', 'float64', 'float64', 'float64']
max_sizes = [1, 1, 1, 1, 1, 1, 3000, 3000, 3000, 3000]


def plot_rechits(X, Y, Z, E, text):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(Z, X, Y, s=np.log(np.log(E + 1)+1) * 100, cmap=cmx.hsv)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    ax.set_title(text)
    plt.show()




inputs_feed = get_input_feeds(args.input)
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while True:
        S, SL, A, L_OH, N = sess.run(list(inputs_feed))
        N = int(N)
        X = S[0:N, 0]
        Y = S[0:N,1]
        Z = S[0:N,2]

        E = A[0:N,3]
        text = find_type(L_OH)
        plot_rechits(X, Y, Z, E, text)
