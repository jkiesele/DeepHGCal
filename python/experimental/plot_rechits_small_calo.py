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
import matplotlib.pyplot as plt


max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to input file")
# parser.add_argument('output', help="Path where to produce output files")
# parser.add_argument("--jobs", default=4, help="Number of processes")
args = parser.parse_args()


def process(input_file):
    file_path = input_file
    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                'isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short']
    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32']
    max_size = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 1, 1, 1, 1, 1, 1]

    data, sizes = sparse_hgcal.read_np_array(file_path, location, branches, types, max_size)


    X = []
    Y = []
    L = []
    for j in range(100):
        x = data[0][j]
        y = data[1][j]
        l = data[6][j]

        x = x[0:sizes[0][j]]
        y = y[0:sizes[1][j]]
        l = l[0:sizes[6][j]]

        X.extend(x)
        Y.extend(y)
        L.extend(l)

    print(np.unique(L))
    print(np.shape(X))

    plt.hist2d(X, Y, bins=(50, 50), cmap=plt.cm.jet)
    plt.show()


process(args.input)
