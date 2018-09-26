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
from libs.prepare_merge_set import load_data_target_center, load_data_target_min_classification
from multiprocessing import Process

max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
parser.add_argument("--type", default=0, help="0 For center of shower as target. 1 for fraction as target.")
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


def worker(data):
    data, num_entries, i, output_file_prefix = data
    write_to_tf_records(data, num_entries, i, output_file_prefix)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def run_conversion_multi_threaded(input_file):
    jobs = int(args.jobs)

    return_dict = load_data_target_center(input_file) if int(args.type) == 0 else load_data_target_min_classification(input_file)
    data_output = return_dict['data_output_merged']
    num_entries_result = return_dict['num_entries_result']

    total_events = len(data_output)

    processes = []
    events_per_jobs = int(total_events / jobs)
    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'
    for i in range(jobs):
        start = i * events_per_jobs
        stop = (i + 1) * events_per_jobs
        D = data_output[start:stop]
        N = num_entries_result[start:stop]

        output_file_prefix = os.path.join(args.output, just_file_name + "_" + str(i) + "_")
        data_packed = D, N, i, output_file_prefix
        processes.append(Process(target=worker, args=(data_packed,)))


    for p in processes:
        p.start()

    for p in processes:
        p.join()


start = time.time()
for item in file_paths:
    run_conversion_multi_threaded(item)
end = time.time()

print("It took", start - end)