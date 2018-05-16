import numpy as np
import root_numpy
import sys
import tensorflow as tf


def euclidean_two_datasets(A, B):
    """
    Returns euclidean distance between two datasets

    A is first dataset in form (N,F) where N is number of examples in first dataset, F is number of features
    B is second dataset in form (M,F) where M is number of examples in second dataset, F is number of features

    Returns:
    A matrix of size (N,M) where each element (i,j) denotes euclidean distance between ith entry in first dataset and jth in second dataset

    """
    A = np.array(A)
    B = np.array(B)
    return np.sqrt(-2*A.dot(B.transpose()) + (np.sum(B*B,axis=1)) + (np.sum(A*A,axis=1))[:,np.newaxis])


def work(A, i, writer, max_entries):
    X = A['rechit_x'][i]
    Y = A['rechit_y'][i]
    Z = A['rechit_z'][i]
    E = A['rechit_energy'][i]

    truth = np.array([int(A['isElectron'][i]), int(A['isMuon'][i]), int(A['isPionCharged'][i]),
                      int(A['isPionNeutral'][i]), int(A['isK0Long'][i]), int(A['isK0Short'][i])],dtype=np.int32)
    assert np.sum(truth) == 1

    if not(len(X) == len(Y) == len(Z) == len(E)) or len(X) == 0:
        print("Error in number of entries")
        return

    num_entries = len(X)

    X = X[0:min(num_entries, 2000)]
    Y = Y[0:min(num_entries, 2000)]
    Z = Z[0:min(num_entries, 2000)]
    E = E[0:min(num_entries, 2000)]

    # N*3 (N = number of entries, 3 is xyz spatial coordinates) matrix
    space_features = np.concatenate((np.expand_dims(X, axis=1),np.expand_dims(Y, axis=1),np.expand_dims(Z, axis=1)), axis=1)

    # Euclidean distance of each entry with each entry
    # It is monotonically increasing function. So it is same as simple absolute difference.
    # TODO: Use broadast subtract op rather than this
    # (i,j)th entry represents euclidean distance between ith and jth entry (symmetric matrix)
    MM = euclidean_two_datasets(space_features, space_features)
    N = np.argsort(MM, axis=1)[:,0:10]
    # D = MM[np.arange(np.shape(N)[0])[:, None], N]

    # All features
    all_features = np.concatenate((np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1), np.expand_dims(Z, axis=1),
                                   np.expand_dims(E, axis=1)), axis=1)

    _all_features = np.zeros((max_entries, 4))
    _all_features[0:num_entries,:] = all_features
    _space_features = np.zeros((max_entries, 3))
    _space_features[0:num_entries,:] = space_features
    _N = np.zeros((max_entries, 10))
    _N[0:num_entries, :] = N

    feature = dict()
    feature['space_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=_space_features.astype(np.float32).flatten()))
    feature['all_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=_all_features.astype(np.float32).flatten()))
    feature['neighbor_matrix'] = tf.train.Feature(int64_list=tf.train.Int64List(value=_N.astype(np.int64).flatten()))
    feature['labels_one_hot'] = tf.train.Feature(int64_list=tf.train.Int64List(value=truth.astype(np.int64).flatten()))
    feature['num_entries'] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([num_entries]).astype(np.int64).flatten()))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def convert(input_file, output_file_prefix):
    branches = []
    branches.append('rechit_x')
    branches.append('rechit_y')
    branches.append('rechit_z')
    branches.append('rechit_layer')
    branches.append('rechit_energy')

    branches.append('isElectron')
    branches.append('isMuon')
    branches.append('isPionCharged')
    branches.append('isPionNeutral')
    branches.append('isK0Long')
    branches.append('isK0Short')

    # Initiating the writer and creating the tfrecords file.

    max_entries = 2000

    n_entires = len(root_numpy.root2array(input_file, branches=['isElectron'])['isElectron'])
    print("Events", n_entires)

    i_entry = 0
    j_entry = 0
    chunk_size = 200
    while j_entry < n_entires:
        writer = tf.python_io.TFRecordWriter(output_file_prefix + str(i_entry) + '.tfrecords',
                                             options=tf.python_io.TFRecordOptions(
                                                 tf.python_io.TFRecordCompressionType.GZIP))
        A = root_numpy.root2array(input_file, branches=branches, treename=None, start=i_entry,
                                  stop=min(i_entry + chunk_size, n_entires))
        for i in range(len(A)):
            work(A, i, writer, max_entries)
            print("Written", j_entry)
            j_entry += 1

        i_entry += chunk_size
        writer.close()

