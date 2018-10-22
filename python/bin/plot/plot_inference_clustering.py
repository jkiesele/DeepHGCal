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


histogram_values_resolution=[]

index=0
with open(os.path.join(config['test_out_path'], 'inference_output_files.txt')) as f:
    content = f.readlines()
    for i in content:
        with gzip.open(i.strip()) as f:
            data=pickle.load(f)
            for j in data:
                input, num_entries, output = j
                output = output[:,0:2]

                spatial = input[:, spatial_features_indices]
                targets = input[:, target_indices]
                energy = input[:,other_features_indices][:,0]

                num_entries = float(np.asscalar(num_entries))


                diff_sq_1 = (output - targets) ** 2 * energy[:, np.newaxis]  # TODO: Multiply by sequence mask
                loss_1 = ((1/num_entries)*np.sum(diff_sq_1) / np.sum(energy, axis=-1)) * float(num_entries!=0)

                diff_sq_2 = (output - (1-targets)) ** 2 * energy[:, np.newaxis]  # TODO: Multiply by sequence mask
                loss_2 = ((1/num_entries)*np.sum(diff_sq_2) / np.sum(energy, axis=-1)) * float(num_entries!=0)

                shower_indices = np.argmin(np.array([loss_1, loss_2]))

                if loss_1 < loss_2:
                    sorted_target = targets
                else:
                    sorted_target = 1-targets

                perf1 = np.sum(output[:,0]*energy) / np.sum(sorted_target[:,0] * energy)
                perf2 = np.sum(output[:,1]*energy) / np.sum(sorted_target[:,1] * energy)

                histogram_values_resolution.append(perf1)
                histogram_values_resolution.append(perf2)

                # a = plt.figure(0)
                # a.suptitle('Output')
                # b = plt.figure(1)
                # b.suptitle('GT')
                # plot_clustering(spatial=spatial, energy=energy, prediction=output, fig=a)
                # plot_clustering(spatial=spatial, energy=energy, prediction=sorted_target, fig=b)
                # print("%05.5f %05.5f %05.5f %05.5f" % (loss_1, loss_2, perf1, perf2))
                #
                # plt.show()

                index+=1


mean = np.mean(histogram_values_resolution)
variance = np.var(histogram_values_resolution)

print("Mean:", mean, "Variance:", variance)

bins = np.linspace(0,3,num=30)
plt.hist(histogram_values_resolution, bins=bins)
plt.ylabel('Frequency');
plt.show()