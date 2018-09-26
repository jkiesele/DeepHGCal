import os
from queue import Queue # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import numpy as np
import tensorflow as tf
import sparse_hgcal
from numba import jit
import time
from multiprocessing import Process


DIM_1 = 20
DIM_2 = 20
DIM_3 = 25
HALF_X = 150
HALF_Y = 150
MAX_ELEMENTS = 6

BIN_WIDTH_X = 2 * HALF_X / DIM_1
BIN_WIDTH_Y = 2 * HALF_Y / DIM_2


@jit(nopython=True)
def find_indices(n, histo, eta_bins, phi_bins, layers):
    # n = np.size(eta_bins)
    indices = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if eta_bins[i] >= 0 and eta_bins[i] < DIM_1 and phi_bins[i] >= 0 and phi_bins[i] < DIM_2 and layers[i] >= 0 and \
                layers[i] < DIM_3:
            index = histo[eta_bins[i], phi_bins[i], layers[i]]
            histo[eta_bins[i], phi_bins[i], layers[i]] += 1
        else:
            index = -1
        indices[i] = index
    indices[indices >= 6] = -1
    return indices

@jit
def process(rechit_layer, rechit_x, rechit_y, rechit_z, rechit_energy, rechit_vxy, rechit_vz, seed_x=0, seed_y=0, seed_z=0):
    x_low_edge = seed_x - HALF_X
    x_diff = rechit_x - seed_x
    x_bins = np.floor((rechit_x - x_low_edge) / BIN_WIDTH_X).astype(np.int32)

    y_low_edge = seed_y - HALF_Y
    y_diff = rechit_y - seed_y
    y_bins = np.floor((rechit_y - y_low_edge) / BIN_WIDTH_Y).astype(np.int32)

    layers = np.minimum(np.floor(rechit_layer), DIM_3).astype(np.int32)

    histogram = np.zeros((DIM_1,DIM_2,DIM_3))
    indices = find_indices(np.size(x_bins), histogram, x_bins, y_bins, layers)

    indices_valid = np.where(indices!=-1)
    bins = indices[indices_valid]
    store_energy = rechit_energy[indices_valid]
    store_x = x_diff[indices_valid]
    store_y = y_diff[indices_valid]
    store_x_bins = x_bins[indices_valid]
    store_y_bins = y_bins[indices_valid]
    store_layers = layers[indices_valid]

    store_z = (rechit_z - seed_z)[indices_valid]
    store_vxy = rechit_vxy[indices_valid]
    store_vz = rechit_vz[indices_valid]

    data_x = np.zeros((DIM_1, DIM_2, DIM_3, 7 * MAX_ELEMENTS))
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+0] = store_energy
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+1] = store_layers
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+2] = store_x
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+3] = store_y
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+4] = store_z
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+5] = store_vxy
    data_x[store_x_bins, store_y_bins, store_layers, bins*7+6] = store_vz

    return data_x


parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def _write_entry(x, labels_one_hot, writer):
    feature = dict()
    feature['x'] = tf.train.Feature(float_list=tf.train.FloatList(value=x.astype(np.float32).flatten()))
    feature['labels_one_hot'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels_one_hot.astype(np.int64).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(rechit_x,rechit_y,rechit_z,rechit_vxy,rechit_xz, rechit_energy, rechit_layer, H, I, i, output_file_prefix):
    writer = tf.python_io.TFRecordWriter(output_file_prefix + '.tfrecords',
                                         options=tf.python_io.TFRecordOptions(
                                             tf.python_io.TFRecordCompressionType.GZIP))

    for i in range(len(rechit_x)):
        start = time.time()

        # print(rechit_layer[i, 0:I[i]])
        x = process(rechit_layer[i, 0:I[i]], rechit_x[i, 0:I[i]], rechit_y[i, 0:I[i]], rechit_z[i, 0:I[i]],
                    rechit_energy[i, 0:I[i]], rechit_vxy[i, 0:I[i]], rechit_xz[i, 0:I[i]])
        # x = rechit_layer[i,0:I[i]]

        _write_entry(x, H[i], writer)

        print("Written", i)

    writer.close()


def worker(data):
    A, B, C, D, E, F, G, H, I, i, output_file_prefix = data
    write_to_tf_records(A, B, C, D, E, F, G, H, I, i, output_file_prefix)



def run_conversion_multi_threaded(input_file):
    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'

    file_path = input_file
    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                'isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short']
    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32']
    max_size = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 1, 1, 1, 1, 1, 1]

    data, sizes = sparse_hgcal.read_np_array(file_path, location, branches, types, max_size)

    ex_data_lables = [np.expand_dims(group, axis=2) for group in [data[7], data[8], data[9], data[10], data[11], data[12]]]

    labels_one_hot = np.concatenate(tuple(ex_data_lables), axis=1)
    num_entries = sizes[0]

    assert int(np.mean(np.sum(labels_one_hot, axis=1))) == 1
    total_events = len(sizes[0])

    jobs = int(args.jobs)
    events_per_jobs = int(total_events/jobs)
    processes = []
    for i in range(jobs):
        start = i*events_per_jobs
        stop = (i+1) * events_per_jobs
        A = data[0][start:stop]
        B = data[1][start:stop]
        C = data[2][start:stop]
        D = data[3][start:stop]
        E = data[4][start:stop]
        F = data[5][start:stop]
        G = data[6][start:stop]

        H = labels_one_hot[start:stop]
        I = num_entries[start:stop]

        output_file_prefix = os.path.join(args.output, just_file_name + "_" + str(i) + "_")
        data_packed = A, B, C, D, E, F, G, H, I, i, output_file_prefix
        processes.append(Process(target=worker, args=(data_packed,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()



for item in file_paths:
    run_conversion_multi_threaded(item)
