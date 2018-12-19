import numpy as np
import os
import sys
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import gzip
import pickle
import configparser as cp
from libs.plots import plot_clustering
from matplotlib import cm
from matplotlib.colors import LogNorm


parser = argparse.ArgumentParser(description='Plot clustering model output')
parser.add_argument('input', help="Path to the config file which was used to train")
parser.add_argument('config', help="Config section within the config file")
args = parser.parse_args()

config_file = cp.ConfigParser()
config_file.read(args.input)
config = config_file[args.config]

spatial_features_indices = tuple([int(x) for x in (config['input_spatial_features_indices']).split(',')])
spatial_features_local_indices = tuple([int(x) for x in (config['input_spatial_features_local_indices']).split(',')])
other_features_indices = tuple([int(x) for x in (config['input_other_features_indices']).split(',')])
target_indices = tuple([int(x) for x in (config['target_indices']).split(',')])


energy_values = []
correct = []

index=0
with open(os.path.join(config['test_out_path'], 'inference_output_files.txt')) as f:
    content = f.readlines()
    for i in content:
        with gzip.open(i.strip()) as f:
            data=pickle.load(f)
            for j in data:
                try:
                    input, num_entries, predicted_location, seed_indices = j
                except ValueError:
                    print("Error!")
                    print("Did you forget putting output_seed_indices_in_inference=1 in config file?")
                    exit(0)

                predicted_location = np.reshape(predicted_location, [2, 3])

                spatial = input[:, spatial_features_indices]
                targets = input[:, target_indices]
                energy = input[:,other_features_indices][:,0]

                shower_sum_energy_1 = np.sum(targets[:, 0] * energy)
                shower_sum_energy_2 = np.sum(targets[:, 1] * energy)


                truth_location = np.array([spatial[seed_indices[0]],spatial[seed_indices[1]]])

                # output is of shape [E,2], could argmax over first dimension, would result in 2D vector
                # target indices is of shape 2D vector
                if np.sum(truth_location - np.flip(predicted_location, axis=0)) > np.sum(truth_location - predicted_location):
                    predicted_location = np.flip(predicted_location, axis=0)

                energy_values.append(shower_sum_energy_1)
                correct.append(np.sum((predicted_location[0] - truth_location[0])**2))
                energy_values.append(shower_sum_energy_2)
                correct.append(np.sum((predicted_location[0] - truth_location[0])**2))

                num_entries = float(np.asscalar(num_entries))

                index+=1



def acc_2d_plot(energy_values, correct):
    nbins=5
    energy_values_1 = energy_values[0::2]
    energy_values_2 = energy_values[1::2]
    min_energy, max_energy = np.min(energy_values), np.max(energy_values)

    energy_values_x = np.linspace(min_energy, max_energy, num=nbins)

    bin_indices_x = np.minimum((energy_values_1 - min_energy)*nbins/(max_energy-min_energy), nbins-1).astype(np.int64)
    bin_indices_y = np.minimum((energy_values_2 - min_energy)*nbins/(max_energy-min_energy), nbins-1).astype(np.int64)

    corr_2d = np.zeros((nbins,nbins), dtype=np.float32)
    variance_2d = np.zeros((nbins,nbins), dtype=np.float32)
    freq_2d = np.zeros((nbins,nbins), dtype=np.int64)

    for i in range(len(energy_values_1)):
        corr_2d[bin_indices_y[i], bin_indices_x[i]] += float(correct[i*2])
        corr_2d[bin_indices_x[i], bin_indices_y[i]] += float(correct[i*2+1])
        freq_2d[bin_indices_y[i], bin_indices_x[i]] += 1
        freq_2d[bin_indices_x[i], bin_indices_y[i]] += 1

    corr_2d /= freq_2d


    corr_2d = np.flip(corr_2d, axis=0)
    freq_2d = np.flip(freq_2d, axis=0)

    return corr_2d, freq_2d, energy_values_x

energy_values = np.array(energy_values)
correct = np.array(correct)

corr_2d, freq_2d, energy_values_x_2d = acc_2d_plot(energy_values, correct)

acc_plot = energy_values/correct


plt.clf()
fig = plt.figure(1)
ax = fig.add_subplot(111)


size = 5
x_start = np.min(energy_values_x_2d)
x_end = np.max(energy_values_x_2d)
y_start = np.min(energy_values_x_2d)
y_end = np.max(energy_values_x_2d)

extent = [x_start, x_end, y_start, y_end]

cax = ax.imshow(corr_2d, interpolation='nearest', extent=extent, vmin=0, vmax=100000)

# Add the text
jump_x = (x_end - x_start) / (2.0 * size)
jump_y = (y_end - y_start) / (2.0 * size)
x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        label = ('%.3E' % corr_2d[size -1- y_index, x_index])
        text_x = x + jump_x
        text_y = y + jump_y
        ax.text(text_x, text_y, label, color='black', ha='center', va='center')


plt.xlabel("Test shower energy")
plt.ylabel("Noise shower energy")
plt.title("Mean l2 square")
cbar = fig.colorbar(cax)
plt.savefig(os.path.join(config['test_out_path'], 'acc_2d_fo_energy.png'))



plt.clf()
fig = plt.figure(1)
cax = plt.imshow(freq_2d, interpolation='nearest', extent=[np.min(energy_values_x_2d), np.max(energy_values_x_2d), np.min(energy_values_x_2d), np.max(energy_values_x_2d)], )
plt.xlabel("Test shower energy")
plt.ylabel("Noise shower energy")
plt.title("Frequency")
cbar = fig.colorbar(cax)
plt.savefig(os.path.join(config['test_out_path'], 'frequency_2d_fo_energy.png'))
