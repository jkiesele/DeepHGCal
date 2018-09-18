import numpy as np
from numba import jit
import experimental_mod.helpers as helpers
import root_numpy

DIM_1 = 13
DIM_2 = 13
DIM_3 = 55
HALF_ETA = 0.2
HALF_PHI = 0.2
MAX_ELEMENTS = 6

BIN_WIDTH_ETA = 2 * HALF_ETA / DIM_1
BIN_WIDTH_PHI = 2 * HALF_PHI / DIM_2

@jit
def find_indices(histo, eta_bins, phi_bins, layers):
    n = np.size(eta_bins)
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


def process(rechit_eta, rechit_phi, rechit_layer, rechit_energy, rechit_time, seed_eta, seed_phi, rechit_total_fraction):
    eta_low_edge = seed_eta - HALF_ETA
    eta_diff = rechit_eta - seed_eta
    eta_bins = np.floor((rechit_eta - eta_low_edge) / BIN_WIDTH_ETA).astype(np.int32)

    phi_low_edge = helpers.delta_angle(seed_phi, HALF_PHI)
    phi_diff = rechit_phi - seed_phi
    phi_bins = np.floor(helpers.delta_angle(rechit_phi, phi_low_edge) / BIN_WIDTH_PHI).astype(np.int32)

    layers = np.minimum(np.floor(rechit_layer) - 1, 54).astype(np.int32)

    histogram = np.zeros((DIM_1,DIM_2,DIM_3))
    indices = find_indices(histogram, eta_bins, phi_bins, layers)
    indices_valid = np.where(indices!=-1)
    store_energy = rechit_energy[indices_valid]
    store_time = rechit_time[indices_valid]
    store_fraction = rechit_total_fraction[indices_valid]
    store_eta = eta_diff[indices_valid]
    store_phi = phi_diff[indices_valid]
    store_eta_bins = eta_bins[indices_valid]
    store_phi_bins = phi_bins[indices_valid]
    store_layers = layers[indices_valid]

    data_x = np.zeros((DIM_1, DIM_2, DIM_3, 24), dtype=np.float32)
    data_y = np.zeros((DIM_1, DIM_2, DIM_3, 6), dtype=np.float32)
    data_x[store_eta_bins, store_phi_bins, store_layers, indices[indices_valid]*4+0] = store_energy
    data_x[store_eta_bins, store_phi_bins, store_layers, indices[indices_valid]*4+1] = store_time
    data_x[store_eta_bins, store_phi_bins, store_layers, indices[indices_valid]*4+2] = store_eta
    data_x[store_eta_bins, store_phi_bins, store_layers, indices[indices_valid]*4+3] = store_phi

    data_y[store_eta_bins, store_phi_bins, store_layers, indices[indices_valid]] = store_fraction
    return data_x, data_y


def convert_file(file_name, treename='deepntuplizer/tree',
                 fn_noise_eta=lambda: np.random.normal(0, 0.02),
                 fn_noise_phi=lambda: np.random.normal(0, 0.02),
                 sample_multiplier=2):
    branches = ['isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy', 'rechit_time',
                'rechit_energy', 'rechit_phi', 'rechit_eta', 'rechit_layer', 'seed_eta', 'seed_phi', 'rechit_total_fraction']

    A = root_numpy.root2array(file_name, treename, branches=branches)

    num_entries = len(A['seed_eta'])
    print("Number of entries is ", num_entries)

    data_array_x = np.zeros((num_entries*sample_multiplier, DIM_1, DIM_2, DIM_3, MAX_ELEMENTS * 4), dtype=np.float32)
    data_array_y = np.zeros((num_entries*sample_multiplier, DIM_1, DIM_2, DIM_3, MAX_ELEMENTS), dtype=np.float32)

    for i in range(num_entries):
        # N rechits per event
        rechit_energy = A['rechit_energy'][i]  # N energy
        rechit_time = A['rechit_time'][i]  # N time
        rechit_eta = A['rechit_eta'][i]
        rechit_phi = A['rechit_phi'][i]
        rechit_layer = A['rechit_layer'][i]

        seed_eta = A['seed_eta'][i]  # scalar eta of the seed
        seed_phi = A['seed_phi'][i]  # scalar phi of the seed

        rechit_total_fraction = A['rechit_total_fraction'][i]

        for j in range(sample_multiplier):
            seed_eta_with_noise = seed_eta + fn_noise_eta()
            seed_phi_with_noise = seed_phi + fn_noise_phi()
            pic3dX, pic3dY = process(rechit_eta, rechit_phi, rechit_layer, rechit_energy, rechit_time,
                                     seed_eta_with_noise, seed_phi_with_noise, rechit_total_fraction)
            data_array_x[i * sample_multiplier + j] = pic3dX
            data_array_y[i * sample_multiplier + j] = pic3dY

    return data_array_x, data_array_y
