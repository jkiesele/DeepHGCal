import argparse

def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Plot clustering model output')
parser.add_argument('input', help="Path to the config file which was used to train")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('output', help="Where to produce the plots")
parser.add_argument('--size',  default=-1, help="If you have loaded size samples (filtered), then stop loading more for performance reason")
# parser.add_argument('--cut', default=-1, help="Fraction to pick the shared part")
#
args = parser.parse_args()
# cut = float(args.cut)
# assert cut == -1 or (0 < cut < 1)

if __name__ != "__main__":
    print("Can't import this file")
    exit(0)

show_3d_figures = False #str2bool(args.figures)

import matplotlib as mpl
if not show_3d_figures:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
import math
import gzip
import pickle
import configparser as cp
from libs.plots import plot_clustering, plot_clustering_4
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.backends.backend_pdf

config_file = cp.ConfigParser()
config_file.read(args.input)
config = config_file[args.config]

spatial_features_indices = tuple([int(x) for x in (config['input_spatial_features_indices']).split(',')])
spatial_features_local_indices = tuple([int(x) for x in (config['input_spatial_features_local_indices']).split(',')])
other_features_indices = tuple([int(x) for x in (config['input_other_features_indices']).split(',')])
target_indices = tuple([int(x) for x in (config['target_indices']).split(',')])

filtered_test_size = int(args.size)


def load_response_values():
    #  For cuts 0.0 (no cut), 0.0001, 0.01, 0.05, 0.1, 0.15, 0.2 respectively
    cuts = [0, 0.0001, 0.01, 0.05, 0.1, 0.15, 0.2]
    response_values = [[], [], [], [], [], [], []]
    energy_values = [[], [], [], [], [], [], []]

    # Notice each of the response array:
    # It will be of shape T*2 where T is the filtered test size. Every two consecutive elements will belong to the same
    # test sample. So, (i+1)th element will be noise shower respective to ith element.

    loaded = 0

    with open(os.path.join(config['test_out_path'], 'inference_output_files.txt')) as f:
        content = f.readlines()
        for i in content:
            if loaded > filtered_test_size:
                break
            with gzip.open(i.strip()) as f:
                data=pickle.load(f)
                for j in data:
                    input, num_entries, output = j
                    # Output is the fraction output of the network [Ex2]
                    output = np.nan_to_num(output[:,0:2])
                    # Spatial coordinates [E,3] (unused)
                    spatial = input[:, spatial_features_indices]
                    # Target fraction [E,2]
                    targets = input[:, target_indices]
                    # Energy [E]
                    energy = input[:,other_features_indices][:,0]
                    # Number of actual entries with deposit (unused)
                    num_entries = float(np.asscalar(num_entries))

                    # Compute loss both swapped and unswapped
                    loss_non_swapped = float(np.sum((output[:, 0] - targets[:, 0]) ** 2 * np.sqrt(energy * targets[:, 0])) / np.sum( np.sqrt(energy * targets[:, 0])))\
                                       + float(np.sum((output[:, 1] - targets[:, 1]) ** 2 * np.sqrt(energy * targets[:, 1])) / np.sum( np.sqrt(energy * targets[:, 1])))

                    loss_swapped = float(np.sum((output[:, 0] - targets[:, 1]) ** 2 * np.sqrt(energy * targets[:, 1])) / np.sum( np.sqrt(energy * targets[:, 1])))\
                                       + float(np.sum((output[:, 1] - targets[:, 0]) ** 2 * np.sqrt(energy * targets[:, 0])) / np.sum( np.sqrt(energy * targets[:, 0])))

                    # Since, for every sensor, sum of both the shower fraction is equal to 1, we can swap by subtracting
                    # from 1.
                    if loss_swapped < loss_non_swapped:
                        targets = 1 - targets

                    # Truth sum of the two showers
                    truth_sum_0 = np.sum(energy * targets[:, 0])
                    truth_sum_1 = np.sum(energy * targets[:, 1])

                    # Only take less than 70 GeV
                    if max(truth_sum_0, truth_sum_1) > 70000:
                        continue

                    if loaded > filtered_test_size:
                        continue
                    if loaded%5000 == 0:
                        print("Loaded", loaded)

                    loaded += 1

                    # Compute and store response values
                    icut = 0
                    for cut in cuts:
                        output_copied = output.copy()
                        energy_copied = energy.copy()
                        energy_copied[targets[:, 0] < cut] = 0
                        energy_copied[targets[:, 0] > (1-cut)] = 0

                        response_0 = np.sum(output_copied[:, 0] * energy_copied) / np.sum(targets[:, 0] * energy_copied)
                        response_1 = np.sum(output_copied[:, 1] * energy_copied) / np.sum(targets[:, 1] * energy_copied)

                        response_values[icut].append(np.nan_to_num(response_0))
                        response_values[icut].append(np.nan_to_num(response_1))

                        energy_values[icut].append(truth_sum_0)
                        energy_values[icut].append(truth_sum_1)

                        icut += 1
    return np.array(cuts, dtype=np.float), np.array(response_values, dtype=np.float), np.array(energy_values, dtype=float)


def compute_var_mean(cuts, response_values, energy_values, noise=False):
    # Divisions for bins
    bin_div = ([5, 10, 20, 30, 40, 50, 60])

    # Mean values against true shower energy
    mean_values = np.zeros((len(cuts), len(bin_div) - 1), dtype=np.float)
    variance_values = np.zeros((len(cuts), len(bin_div) - 1), dtype=np.float)

    icut = 0
    for cut in cuts:
        response_values_cut = response_values[icut]

        # If you want to find values against the noise shower, simply flip every two elements of response array
        # since then for every response value, you'll get the energy of the opposite shower
        if noise:
            test_size = int(len(response_values_cut) / 2)
            response_values_cut = np.reshape(response_values_cut, (test_size, 2)) # (01)(01)(01)(01)
            response_values_cut = (response_values_cut[:, 1][..., np.newaxis], response_values_cut[:, 0][..., np.newaxis]) # (10)(10)(10)(10)
            response_values_cut = np.concatenate(response_values_cut, axis=1).flatten() # 10101010

        bin_indices = np.digitize(energy_values[icut], bins=bin_div)
        mean_values_cut = []
        var_values_cut = []

        for div in range(len(bin_div)-1):
            all_corresponding_values = response_values_cut[np.argwhere(bin_indices==div)]
            mean_value = np.mean(all_corresponding_values)
            variance_value = np.var(all_corresponding_values)
            mean_values_cut.append(mean_value)
            var_values_cut.append(variance_value)
            print(mean_value, variance_value)

        mean_values[icut] = mean_values_cut
        variance_values[icut] = var_values_cut
        icut += 1

    return mean_values, variance_values


def make_plot(cuts, values, x_axis_text, y_axis_text):
    plt.gcf()
    fig = plt.figure()
    fig.set_size_inches(10, 7)
    print("Making %d plots"%len(values))
    for value in values:
        plt.plot([10, 20, 30, 40, 50, 60], value)
    plt.xlabel(x_axis_text)
    plt.ylabel(y_axis_text)
    plt.legend(["%f-%f" % (x, 1-x) for x in cuts])


def main():
    # [C], [Cx(T*2)], [Cx(T*2]
    cuts, response_values, energy_values = load_response_values()
    # Convert from MeV to GeV
    energy_values = energy_values / 1000

    mean_values_noise, variance_values_noise = compute_var_mean(cuts, response_values, energy_values, noise=True)
    mean_values_true, variance_values_true = compute_var_mean(cuts, response_values, energy_values)


    make_plot(cuts, mean_values_true, "Test shower energy (GeV)", 'Response (mean)')
    plt.savefig(os.path.join(args.output, "mean_true.pdf"))
    make_plot(cuts, variance_values_true, "Test shower energy (GeV)", 'Response (variance)')
    plt.savefig(os.path.join(args.output, "variance_true.pdf"))
    make_plot(cuts, mean_values_noise, "Noise shower energy (GeV)", 'Response (mean)')
    plt.savefig(os.path.join(args.output, "mean_noise.pdf"))
    make_plot(cuts, variance_values_noise, "Noise shower energy (GeV)", 'Response (variance)')
    plt.savefig(os.path.join(args.output, "variance_noise.pdf"))


if __name__ == '__main__':
    main()

