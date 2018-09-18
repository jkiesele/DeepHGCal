import os
import sys
from queue import Queue # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import uproot


parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to file which should contain full paths of all the root files on separate lines")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--jobs", default=4, help="Number of processes")
args = parser.parse_args()

with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


def _write_entry(all_features, labels_one_hot, num_entries, writer):
    feature = dict()
    feature['all_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=all_features.astype(np.float32).flatten()))
    feature['labels_one_hot'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels_one_hot.astype(np.int64).flatten()))
    feature['num_entries'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([num_entries]).astype(np.int64).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_records(all_features, labels_one_hot, num_entries, i, output_file_prefix):
    writer = tf.python_io.TFRecordWriter(output_file_prefix + '.tfrecords',
                                         options=tf.python_io.TFRecordOptions(
                                             tf.python_io.TFRecordCompressionType.GZIP))

    for i in range(len(all_features)):
        if np.sum(num_entries[i]) != 0 and np.sum(labels_one_hot[i]) == 1:
            _write_entry(all_features[i], labels_one_hot[i], num_entries[i], writer)
        print("Written", i)

    writer.close()


def worker():
    global jobs_queue
    while True:
        all_features, labels_one_hot, num_entries, i, output_file_prefix = jobs_queue.get()

        write_to_tf_records(all_features, labels_one_hot, num_entries, i, output_file_prefix )

        jobs_queue.task_done()


jobs_queue = Queue()
jobs = int(args.jobs)
for i in range(jobs):
    t = Thread(target=worker)
    t.daemon = True
    t.start()


def read_array_from_file(file_path):
    data = []
    sizes = []

    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_eta', 'rechit_phi', 'rechit_energy', 'rechit_layer','rechit_time',
                'isElectron', 'isGamma', 'isMuon', 'isPionCharged']

    max_size = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 1, 1, 1, 1]


    the_tree = uproot.open(file_path)["deepntuplizer"]["tree"]
    data_branches = the_tree.arrays(tuple(branches), outputtype=tuple)

    events = len(data_branches[0])

    for i in range(8):
        data.append(np.zeros((events, max_size[i]), dtype=np.float32))

    for i in range(4):
        data.append(np.zeros((events), dtype=np.int32))

    for i in range(12):
        sizes.append([])

    event_no = 0
    for data_in in zip(data_branches[0], data_branches[1], data_branches[2], data_branches[3]
            , data_branches[4], data_branches[5], data_branches[6], data_branches[7]
            , data_branches[8], data_branches[9], data_branches[10], data_branches[11]):

        for i in range(8):
            sizes[i].append(np.size(data_in[i]))
            data[i][event_no][0:min(len(data_in[i]), max_size[i])] = data_in[i][0:min(len(data_in[i]), max_size[i])]

        for i in range(4):
            sizes[i+8].append(1)
            data[i+8][event_no] = data_in[i+8]

        event_no += 1


    return data, sizes


def run_conversion_multi_threaded(input_file):
    global jobs_queue

    just_file_name = os.path.splitext(os.path.split(input_file)[1])[0] + '_'

    data, sizes = read_array_from_file(input_file)

    ex_data = [np.expand_dims(data[x], axis=2) for x in range(8)]
    ex_data_lables = [np.expand_dims(data[x+8], axis=2) for x in range(4)]

    all_features = np.concatenate(tuple([ex_data[i] for i in range(len(ex_data))]), axis=2)
    labels_one_hot = np.concatenate(tuple(ex_data_lables), axis=1)
    num_entries = sizes[0]

    print(np.sum(labels_one_hot))
    total_events = len(sizes[0])
    assert np.array_equal(np.shape(all_features), [total_events, 3000, 8])

    events_per_jobs = int(total_events/jobs)
    for i in range(jobs):
        start = i*events_per_jobs
        stop = (i+1) * events_per_jobs
        A = all_features[start:stop]
        B = labels_one_hot[start:stop]
        C = num_entries[start:stop]

        output_file_prefix = os.path.join(args.output, just_file_name + "_" + str(i)+"_")
        data_packed = A, B, C, i, output_file_prefix
        jobs_queue.put(data_packed)

    jobs_queue.join()


for item in file_paths:
    run_conversion_multi_threaded(item)
