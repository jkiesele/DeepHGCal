import os
import sys
from queue import Queue  # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import time
from sklearn.utils import shuffle
import sparse_hgcal
import numpy as np
from numba import jit
import experimental_mod.helpers as helpers
import matplotlib.pyplot as plt

DIM_1 = 20
DIM_2 = 20
DIM_3 = 20
HALF_X = 150
HALF_Y = 150
MAX_ELEMENTS = 16

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
    indices[indices >= MAX_ELEMENTS] = -1
    return indices

@jit
def process(rechit_layer, rechit_x, rechit_y, seed_x=0, seed_y=0, seed_z=0):
    x_low_edge = seed_x - HALF_X
    x_diff = rechit_x - seed_x
    x_bins = np.floor((rechit_x - x_low_edge) / BIN_WIDTH_X).astype(np.int32)

    y_low_edge = seed_y - HALF_Y
    y_diff = rechit_y - seed_y
    y_bins = np.floor((rechit_y - y_low_edge) / BIN_WIDTH_Y).astype(np.int32)

    layers = np.minimum(np.floor(rechit_layer), DIM_3).astype(np.int32)

    histogram = np.zeros((DIM_1,DIM_2,DIM_3))
    indices = find_indices(np.size(x_bins), histogram, x_bins, y_bins, layers)
    # np.set_printoptions(threshold=np.nan)

    print("Xbins")
    print(repr(x_bins))
    print("Ybins")
    print(repr(y_bins))
    print("Lbins")
    print(repr(layers))
    print("D")
    print(repr(indices))
    plt.figure()
    plt.title('xbins')
    plt.hist(x_bins,bins=np.alen(np.unique(x_bins)))
    plt.figure()
    plt.title('ybins')
    plt.hist(y_bins,bins=np.alen(np.unique(y_bins)))
    plt.figure()
    plt.title('layers')
    plt.hist(layers,bins=np.alen(np.unique(layers)))
    plt.figure()
    plt.title('indices')
    plt.hist(indices,bins=np.alen(np.unique(indices)))
    plt.show()




max_gpu_events = 500

parser = argparse.ArgumentParser(description='Find indices')
parser.add_argument('input',
                    help="Path to root file")
args = parser.parse_args()

def blabla(input_file):
    print(input_file)
    np.set_printoptions(threshold=np.nan)

    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                'true_x', 'true_y', 'true_r', 'true_energy', 'isGamma']

    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
             'float64', 'float64', 'int32']

    max_size = [2102 for _ in range(7)] + [1, 1, 1, 1, 1]

    nparray, sizes = sparse_hgcal.read_np_array(input_file, location, branches, types, max_size)

    # from root_numpy import root2array
    # nparray = root2array(input_file,
    #                      treename="B4",
    #                      branches=branches,
    #                      # stop=5
    #                      ).view(np.ndarray)


    l = nparray[6][0]
    x = nparray[0][0]
    y = nparray[1][0]

    process(l, x, y)


blabla(args.input)
