import os
import sys
from queue import Queue # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import sparse_hgcal


max_gpu_events = 500


parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def _write_entry(all_features, space_features, spatial_local, labels_one_hot, num_entries, writer):
    feature = dict()
    feature['spatial_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=space_features.astype(np.float32).flatten()))
    feature['spatial_local_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=spatial_local.astype(np.float32).flatten()))
    feature['all_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=all_features.astype(np.float32).flatten()))
    feature['labels_one_hot'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels_one_hot.astype(np.int64).flatten()))
    feature['num_entries'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([num_entries]).astype(np.int64).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(all_features, space_features, spatial_local, labels_one_hot, num_entries, i, output_file_prefix):
    writer = tf.python_io.TFRecordWriter(output_file_prefix + '.tfrecords',
                                         options=tf.python_io.TFRecordOptions(
                                             tf.python_io.TFRecordCompressionType.GZIP))
    for i in range(len(all_features)):
        _write_entry(all_features[i], space_features[i], spatial_local[i], labels_one_hot[i], num_entries[i], writer)
        print("Written", i)

    writer.close()


def worker():
    global jobs_queue
    while True:
        all_features, space_features, spatial_local, labels_one_hot, num_entries, i, output_file_prefix = jobs_queue.get()

        write_to_tf_records(all_features, space_features, spatial_local, labels_one_hot, num_entries, i, output_file_prefix )

        jobs_queue.task_done()

jobs_queue = Queue()
jobs = int(args.jobs)
for i in range(jobs):
    t = Thread(target=worker)
    t.daemon = True
    t.start()


def run_conversion_multi_threaded(input_file):
    global jobs_queue, max_gpu_events

    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'

    file_path = input_file
    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                'isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short']
    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32']
    max_size = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 1, 1, 1, 1, 1, 1]

    data, sizes = sparse_hgcal.read_np_array(file_path, location, branches, types, max_size)

    ex_data = [np.expand_dims(group, axis=2) for group in [data[0], data[1], data[2], data[3], data[4], data[5], data[6]]]
    ex_data_lables = [np.expand_dims(group, axis=2) for group in [data[7], data[8], data[9], data[10], data[11], data[12]]]

    all_features = np.concatenate((ex_data[0], ex_data[1], ex_data[2], ex_data[5], ex_data[6]), axis=2)
    spatial = np.concatenate((ex_data[0], ex_data[1], ex_data[2]), axis=2)
    spatial_local = np.concatenate((ex_data[3], ex_data[4]), axis=2)

    labels_one_hot = np.concatenate(tuple(ex_data_lables), axis=1)
    num_entries = sizes[0]

    assert int(np.mean(np.sum(labels_one_hot, axis=1))) == 1
    total_events = len(sizes[0])
    assert np.array_equal(np.shape(all_features), [total_events, 3000, 5])
    assert np.array_equal(np.shape(spatial), [total_events, 3000, 3])
    assert np.array_equal(np.shape(spatial_local), [total_events, 3000, 2])

    events_per_jobs = int(total_events/jobs)
    for i in range(jobs):
        start = i*events_per_jobs
        stop = (i+1) * events_per_jobs
        A = all_features[start:stop]
        B = spatial[start:stop]
        C = spatial_local[start:stop]
        D = labels_one_hot[start:stop]
        E = num_entries[start:stop]

        output_file_prefix = os.path.join(args.output, just_file_name + "_" + str(i)+"_")
        data_packed = A,B,C,D,E, i, output_file_prefix
        jobs_queue.put(data_packed)

    jobs_queue.join()


for item in file_paths:
    run_conversion_multi_threaded(item)
