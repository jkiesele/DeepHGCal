import numpy as np
import os
import sys
import argparse
import matplotlib as mpl
# mpl.use('Agg')
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
histogram_values_resolution=[]
loss_values = []

index=0
with open(os.path.join(config['test_out_path'], 'inference_output_files.txt')) as f:
    content = f.readlines()
    for i in content:
        with gzip.open(i.strip()) as f:
            data=pickle.load(f)
            for j in data:
                input, num_entries, output = j
                output = np.nan_to_num(output[:,0:2])

                spatial = input[:, spatial_features_indices]
                targets = input[:, target_indices]
                energy = input[:,other_features_indices][:,0]

                num_entries = float(np.asscalar(num_entries))

                diff_sq_1 = (output - targets) ** 2 * energy[:, np.newaxis]  # TODO: Multiply by sequence mask
                loss_1 = ((1/num_entries)*np.sum(diff_sq_1) / np.sum(energy, axis=-1)) * float(num_entries!=0)

                diff_sq_2 = (output - (1-targets)) ** 2 * energy[:, np.newaxis]  # TODO: Multiply by sequence mask
                loss_2 = ((1/num_entries)*np.sum(diff_sq_2) / np.sum(energy, axis=-1)) * float(num_entries!=0)

                shower_indices = np.argmin(np.array([loss_1, loss_2]))

                loss_values.append(min(loss_1, loss_2))

                if loss_1 < loss_2:
                    sorted_target = targets
                else:
                    sorted_target = 1-targets

                perf1 = np.sum(output[:,0]*energy) / np.sum(sorted_target[:,0] * energy)
                perf2 = np.sum(output[:,1]*energy) / np.sum(sorted_target[:,1] * energy)

                truth_energy_sum_1 = np.sum(sorted_target[:,0] * energy)
                truth_energy_sum_2 = np.sum(sorted_target[:,1] * energy)

                histogram_values_resolution.append(perf1)
                energy_values.append(truth_energy_sum_1)
                histogram_values_resolution.append(perf2)
                energy_values.append(truth_energy_sum_2)

                # a = plt.figure(0)
                # a.suptitle('Output')
                # b = plt.figure(1)
                # b.suptitle('GT')
                # plot_clustering(spatial=spatial, energy=energy, prediction=output, fig=a)
                # plot_clustering(spatial=spatial, energy=energy, prediction=sorted_target, fig=b)
                # print("%05.5f %05.5f %05.5f %05.5f" % (loss_1, loss_2, perf1, perf2))
                # plt.show()

                index+=1


mean = np.mean(histogram_values_resolution)
variance = np.var(histogram_values_resolution)
loss_mean = np.mean(loss_values)
loss_variance = np.var(loss_values)

print("X-Test", np.max(energy_values), np.min(energy_values))
print("Y-Test", max(energy_values), np.min(energy_values))


def get_mean_variance_histograms(energy_values, histogram_values_resolution):
    nbins=20
    min_value = np.min(energy_values)
    max_value = np.max(energy_values)

    bin_index = np.minimum((energy_values - min_value)*nbins/(max_value-min_value), nbins-1).astype(np.int64)

    energy_values_x = np.linspace(min_value, max_value, num=nbins)

    resolution_sum = np.zeros(nbins, dtype=np.float64)
    square_difference = np.zeros(nbins, dtype=np.float64)
    count = np.zeros(nbins, dtype=np.int64)

    for i in range(len(energy_values)):
        resolution_sum[bin_index[i]] += histogram_values_resolution[i]
        count[bin_index[i]] += 1

    mean_resolution_values = resolution_sum/count
    # mean_resolution_values[count==0]=0

    for i in range(len(energy_values)):
        square_difference[bin_index[i]] += float(histogram_values_resolution[i] - mean_resolution_values[bin_index[i]])**2

    varaince_resolution_values = square_difference / (count)

    # mean_resolution_values[(count-1)==0]=0
    # mean_resolution_values[(count)==0]=0

    return mean_resolution_values, varaince_resolution_values, energy_values_x, count


def diff_2d_plot(energy_values, histogram_values_resolution):
    nbins=10
    energy_values_1 = energy_values[0::2]
    energy_values_2 = energy_values[1::2]
    min_energy, max_energy = np.min(energy_values), np.max(energy_values)

    energy_values_x = np.linspace(min_energy, max_energy, num=nbins)

    bin_indices_x = np.minimum((energy_values_1 - min_energy)*nbins/(max_energy-min_energy), nbins-1).astype(np.int64)
    bin_indices_y = np.minimum((energy_values_2 - min_energy)*nbins/(max_energy-min_energy), nbins-1).astype(np.int64)

    mean_2d = np.zeros((nbins,nbins), dtype=np.float32)
    variance_2d = np.zeros((nbins,nbins), dtype=np.float32)
    freq_2d = np.zeros((nbins,nbins), dtype=np.int64)

    for i in range(len(energy_values_1)):
        mean_2d[bin_indices_y[i], bin_indices_x[i]] += histogram_values_resolution[i*2]
        mean_2d[bin_indices_x[i], bin_indices_y[i]] += histogram_values_resolution[i*2+1]
        freq_2d[bin_indices_y[i], bin_indices_x[i]] += 1
        freq_2d[bin_indices_x[i], bin_indices_y[i]] += 1

    mean_2d /= freq_2d

    for i in range(len(energy_values_1)):
        variance_2d[bin_indices_y[i], bin_indices_x[i]] += float(histogram_values_resolution[i*2] - mean_2d[bin_indices_y[i], bin_indices_x[i]])**2
        variance_2d[bin_indices_x[i], bin_indices_y[i]] += float(histogram_values_resolution[i*2+1] - mean_2d[bin_indices_x[i], bin_indices_y[i]])**2

    variance_2d = variance_2d/(freq_2d)

    mean_2d[freq_2d == 0] = 0
    variance_2d[freq_2d == 0] = 0

    mean_2d = np.flip(mean_2d, axis=0)
    freq_2d = np.flip(freq_2d, axis=0)
    variance_2d = np.flip(variance_2d, axis=0)


    return mean_2d, variance_2d, freq_2d, energy_values_x


resolution_mean_fo_energy, resolution_variance_fo_energy, energy_values_x, count = get_mean_variance_histograms(energy_values, histogram_values_resolution)
mean_2d, variance_2d, count_2d, energy_values_x_2d = diff_2d_plot(energy_values, histogram_values_resolution)

accuracy = float(np.sum((np.array(histogram_values_resolution)>0.7) & (np.array(histogram_values_resolution)<1.3)))/float(np.size(np.array(histogram_values_resolution)))
variance_from_1 = np.mean((np.array(histogram_values_resolution)-1)**2)

output_string = str(("Resolution mean:", mean, "Resolution variance :", variance, "Loss mean:", loss_mean, "Loss variance:", loss_variance, "Accuracy", accuracy, "Variance from 1", variance_from_1))

print("Samples tested", np.alen(histogram_values_resolution)/2)
print(output_string)

bins = np.linspace(-0.1,3.1,num=32)
histogram_values_resolution_2 = np.copy(histogram_values_resolution)
histogram_values_resolution_2[histogram_values_resolution_2<0.2] = -0.05
histogram_values_resolution_2[histogram_values_resolution_2>2.8] = 3.05
plt.hist(histogram_values_resolution_2, bins=bins)
plt.ylabel('Frequency')
plt.xlabel("Resolution")
# plt.show()
plt.savefig(os.path.join(config['test_out_path'], 'resolution_histogram.png'))

plt.clf()
plt.plot(energy_values_x, count)
plt.xlabel("Energy")
plt.ylabel('Frequency')
# plt.show()
plt.savefig(os.path.join(config['test_out_path'], 'energy_histogram.png'))



plt.clf()

plt.subplot(2, 1, 1)
plt.plot(energy_values_x, resolution_mean_fo_energy)
plt.xlabel("Energy")
plt.ylabel('Resolution (mean)')
# plt.show()

plt.subplot(2, 1, 2)
plt.plot(energy_values_x, resolution_variance_fo_energy)
plt.xlabel("Energy")
plt.ylabel('Resolution (variance)')
# plt.show()
plt.savefig(os.path.join(config['test_out_path'], 'fo_energy.png'))


plt.clf()
fig = plt.figure(1)
cax = plt.imshow(mean_2d, interpolation='nearest', extent=[np.min(energy_values_x_2d), np.max(energy_values_x_2d), np.min(energy_values_x_2d), np.max(energy_values_x_2d)])
plt.xlabel("Shower 1 energy")
plt.ylabel("Shower 2 energy")
plt.title("Response of shower 1 (mean)")
cbar = fig.colorbar(cax)
plt.savefig(os.path.join(config['test_out_path'], 'response_mean_2d_fo_energy.png'))


plt.clf()
fig = plt.figure(1)
cax = plt.imshow(variance_2d, interpolation='nearest', extent=[np.min(energy_values_x_2d), np.max(energy_values_x_2d), np.min(energy_values_x_2d), np.max(energy_values_x_2d)])
plt.xlabel("Shower 1 energy")
plt.ylabel("Shower 2 energy")
plt.title("Response of shower 1 (variance)")
cbar = fig.colorbar(cax)
plt.savefig(os.path.join(config['test_out_path'], 'response_variance_2d_fo_energy.png'))


plt.clf()
fig = plt.figure(1)
cax = plt.imshow(count_2d, interpolation='nearest', extent=[np.min(energy_values_x_2d), np.max(energy_values_x_2d), np.min(energy_values_x_2d), np.max(energy_values_x_2d)], )
plt.xlabel("Shower 1 energy")
plt.ylabel("Shower 2 energy")
plt.title("Frequency")
cbar = fig.colorbar(cax)
plt.savefig(os.path.join(config['test_out_path'], 'frequency_2d_fo_energy.png'))

histogram_values_resolution_3 = np.copy(histogram_values_resolution)
histogram_values_resolution_3 = histogram_values_resolution_3[histogram_values_resolution_3<=2.5]
histogram_values_resolution_3 = histogram_values_resolution_3[histogram_values_resolution_3 >=0.05]
output_string_3 = str(("Inlier resolution mean:", np.mean(histogram_values_resolution_3), "Inlier resolution variance :",
                       np.var(histogram_values_resolution_3),
                      "Efficiency", str(np.alen(histogram_values_resolution_3)/float(np.alen(histogram_values_resolution)))))
print(output_string_3)



with open(os.path.join(config['test_out_path'], 'test_summary.txt'), "w") as text_file:
    text_file.write(output_string)