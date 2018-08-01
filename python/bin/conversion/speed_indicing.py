import numpy as np
from numba import jit

DIM_1 = 20
DIM_2 = 20
DIM_3 = 25
HALF_X = 150
HALF_Y = 150
MAX_ELEMENTS = 6

BIN_WIDTH_X = 2 * HALF_X / DIM_1
BIN_WIDTH_Y = 2 * HALF_Y / DIM_2


@jit(nopython=True, parallel=True)
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
