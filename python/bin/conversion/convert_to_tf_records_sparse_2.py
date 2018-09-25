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


def _write_entry(rechit_data, labels_one_hot, num_entries, writer):
    feature = dict()
    feature['rechit_data'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=rechit_data.astype(np.float32).flatten()))
    feature['labels_one_hot'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=labels_one_hot.astype(np.int64).flatten()))
    feature['num_entries'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.array([num_entries]).astype(np.int64).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(rechit_data, labels_one_hot, num_entries,
                        output_file_prefix):
    writer = tf.python_io.TFRecordWriter(output_file_prefix + '.tfrecords',
                                         options=tf.python_io.TFRecordOptions(
                                             tf.python_io.TFRecordCompressionType.GZIP))
    for i in range(len(rechit_data)):
        _write_entry(rechit_data[i], labels_one_hot[i], num_entries[i], writer)
        print("Written", i)

    writer.close()


def worker():
    global jobs_queue
    while True:
        rechit_data, labels_one_hot, num_entries, output_file_prefix = jobs_queue.get()
        write_to_tf_records(rechit_data, labels_one_hot, num_entries, output_file_prefix)

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
                'isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short']
    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32']
    max_size = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 1, 1, 1, 1, 1, 1]

    data, sizes = sparse_hgcal.read_np_array(file_path, location, branches, types, max_size)

    per_rechit_data = np.concatenate([np.expand_dims(group, axis=2) for group in
               [data[0], data[1], data[2], data[3], data[4], data[5], data[6]]], axis=2)

    labels_one_hot = np.concatenate([np.expand_dims(group, axis=2) for group in [data[7], data[8], data[9], data[10], data[11], data[12]]], axis=1)

    num_entries = sizes[0]

    if args.pion_vs_electron:
        labels_indexed = np.argmax(labels_one_hot, axis=1)
        interesting_indices = np.where((labels_indexed == 0) + (labels_indexed == 3))

        labels_indexed = labels_indexed[interesting_indices]
        labels_indexed[labels_indexed==3] = 1

        labels_one_hot = one_hot(labels_indexed, num_classes=2)

        per_rechit_data = data[interesting_indices]
        num_entries = num_entries[interesting_indices]


    total_events = len(labels_one_hot)

    assert int(np.mean(np.sum(labels_one_hot, axis=1))) == 1
    assert np.array_equal(np.shape(per_rechit_data), [total_events, 3000, 7])

    events_per_jobs = int(total_events / jobs)
    for i in range(jobs):
        start = i * events_per_jobs
        stop = (i + 1) * events_per_jobs
        per_rechit_data_job = per_rechit_data[start:stop]
        labels_jobs = labels_one_hot[start:stop]
        num_entries_jobs = num_entries[start:stop]

        output_file_prefix = os.path.join(args.output, just_file_name + "_" + str(i) + "_")
        data_packed = per_rechit_data_job, labels_jobs, num_entries_jobs, output_file_prefix
        jobs_queue.put(data_packed)

    jobs_queue.join()


for item in file_paths:
    run_conversion_multi_threaded(item)
