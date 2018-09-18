import os
import sys
from queue import Queue  # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import sparse_hgcal
from libs.merge import merge_two_arrays, merge_two_arrays_separate
import time

max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
parser.add_argument("--pion-vs-electron", default=False, help="Whether to only keep pions and electrons")
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def _write_entry(data, num_entries, writer):
    feature = dict()
    feature['data'] = tf.train.Feature( # TODO: Fix this
        float_list=tf.train.FloatList(value=data.astype(np.float32).flatten()))
    feature['num_entries'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.array([num_entries]).astype(np.int64).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(data, num_entries, i, output_file_prefix):
    writer = tf.python_io.TFRecordWriter(output_file_prefix + '.tfrecords',
                                         options=tf.python_io.TFRecordOptions(
                                             tf.python_io.TFRecordCompressionType.GZIP))
    for i in range(len(data)):
        _write_entry(data[i], num_entries[i], writer)
        print("Written", i)

    writer.close()


def worker():
    global jobs_queue
    while True:
        data, num_entries, i, output_file_prefix = jobs_queue.get()
        write_to_tf_records(data, num_entries, i, output_file_prefix)
        jobs_queue.task_done()


jobs_queue = Queue()
jobs = int(args.jobs)
for i in range(jobs):
    t = Thread(target=worker)
    t.daemon = True
    t.start()


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def run_conversion_multi_threaded(input_file):
    global jobs_queuze, max_gpu_events

    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'

    file_path = input_file
    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                'rechit_detid', 'isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short']

    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int32',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32']

    max_size = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 1, 1, 1, 1, 1, 1]

    data, sizes = sparse_hgcal.read_np_array(file_path, location, branches, types, max_size)

    ex_ids = data[7]

    ex_data = [np.expand_dims(group, axis=2) for group in
               [data[0], data[1], data[2], data[3], data[4], data[5], data[6]]]
    ex_data_lables = [np.expand_dims(group, axis=2) for group in
                      [data[7], data[8], data[9], data[10], data[11], data[12]]]
    ex_sizes = sizes[0]

    other_features = np.concatenate((ex_data[5], ex_data[6]), axis=2)
    spatial = np.concatenate((ex_data[0], ex_data[1], ex_data[2]), axis=2)
    spatial_local = np.concatenate((ex_data[3], ex_data[4]), axis=2)

    num_entries = sizes[0]

    indices_1 = np.arange(len(data[0]))
    np.random.shuffle(indices_1)

    indices_2 = np.arange(len(data[0]))
    np.random.shuffle(indices_2)

    pairs = np.concatenate((np.expand_dims(indices_1, axis=1),
                           np.expand_dims(indices_2, axis=1)), axis=1)

    output_data_all = list()

    for i, j in pairs:
        if i == j:
            continue

        ids_1 = ex_ids[i].astype(np.int32) # [3000]
        features_others_1 = other_features[i]
        energy_1 = features_others_1[:,0]
        features_spatial_local_1 = spatial_local[i]
        features_spatial_1 = spatial[i] # VxF

        features_combined_1 = np.concatenate((features_others_1, features_spatial_1, features_spatial_local_1), axis=1).astype(np.float32)
        sizes_1 = ex_sizes[i].astype(np.int32)
        location_1 = np.sum(features_spatial_1 * energy_1[..., np.newaxis], axis=0) / np.sum(energy_1)

        ids_2 = ex_ids[j].astype(np.int32)
        features_others_2 = other_features[j]
        energy_2 = (other_features[j])[:,0]
        features_spatial_local_2 = spatial_local[j]
        features_spatial_2 = spatial[j]
        features_combined_2 = np.concatenate((features_others_2, features_spatial_2, features_spatial_local_2), axis=1).astype(np.float32)
        sizes_2 = ex_sizes[j].astype(np.int32)
        location_2 = np.sum(features_spatial_2 * energy_2[..., np.newaxis], axis=0) / np.sum(energy_2)

        output_data_all.append((features_combined_1, features_combined_2, ids_1, ids_2, sizes_1, sizes_2, location_1, location_2))


    features_1 = np.concatenate([(x[0])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.float32)
    features_2 = np.concatenate([(x[1])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.float32)
    ids_1 = np.concatenate([(x[2])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.int32)
    ids_2 = np.concatenate([(x[3])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.int32)
    sizes_1 = np.concatenate([(x[4])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.int32)
    sizes_2 = np.concatenate([(x[5])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.int32)
    locations_1 = np.concatenate([(x[6])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.float32)
    locations_2 = np.concatenate([(x[7])[np.newaxis, ...] for x in output_data_all], axis=0).astype(np.float32)

    merged_features, _, num_entries_result = merge_two_arrays_separate(
        features_1,
        features_2,
        ids_1,
        ids_2,
        sizes_1,
        sizes_2,
    )


    energies_1 = merged_features[:, :, 0]
    energies_2 = merged_features[:, :, 1]
    e = energies_1 + energies_2

    fractions = energies_1/np.ma.masked_array(e, mask=e==0) # 10000x6000

    target = fractions[..., np.newaxis] * locations_1[:, np.newaxis, :] + (1 - fractions[..., np.newaxis]) * locations_2[:, np.newaxis, :]

    # target[e==0, :] = 0
    target = np.array(target)

    merged_features[:, :, 1] += merged_features[:, :, 0]
    merged_features = merged_features[:, :, 1:]

    data_output = np.concatenate((merged_features, target), axis=2).astype(np.float32)

    total_events = len(target)

    assert np.array_equal(np.shape(merged_features[0]), [6000, 7])
    assert np.array_equal(np.shape(target[0]), [6000, 3])
    assert len(target) == len(num_entries_result) and len(target) == len(merged_features)

    events_per_jobs = int(total_events / jobs)
    for i in range(jobs):
        start = i * events_per_jobs
        stop = (i + 1) * events_per_jobs
        D = data_output[start:stop]
        N = num_entries_result[start:stop]

        output_file_prefix = os.path.join(args.output, just_file_name + "_" + str(i) + "_")
        data_packed = D, N, i, output_file_prefix
        jobs_queue.put(data_packed)

    jobs_queue.join()


for item in file_paths:
    run_conversion_multi_threaded(item)
