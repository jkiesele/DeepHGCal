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
parser.add_argument('input', help="Path to the directory containing plot data")
parser.add_argument('output', help="Path to to the directory to produce plots")
parser.add_argument('--figures', help="Whether to show 3d plots", default=False)
args = parser.parse_args()

if __name__ != "__main__":
    print("Can't import this file")
    exit(0)

show_3d_figures = str2bool(args.figures)


import matplotlib as mpl
# from matplotlib import rc
# rc('text', usetex=True)
if not show_3d_figures:
    mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import gzip
import pickle
import numpy as np
plt.rcParams.update({'font.size': 22})
import matplotlib.backends.backend_pdf



Rs = []
Es = []

colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
colors_2 = []

models_names = []



for p in os.listdir(args.input):
    if not p.endswith('.pbin'):
        continue
    if 'single_neighbours_plusmean' in p:
        models_names.append('GravNet')
        colors_2.append(colors[1])
    elif 'hidden_aggregators_plusmean' in p:
        models_names.append('GarNet')
        colors_2.append(colors[3])
    elif 'dgcnn' in p:
        models_names.append('DGCNN')
        colors_2.append(colors[0])
    elif 'output_binning20_2' in p:
        models_names.append('Binning')
        colors_2.append(colors[2])
    else:
        0/0


    full_path = os.path.join(args.input, p)

    with gzip.open(os.path.join(full_path), 'rb') as f:
        R, E = pickle.load(f)

    R = np.reshape(R, (int(np.alen(R)/2), 2))
    print(np.shape(R[:, 1][...,np.newaxis]))
    print(np.shape(R[:, 0][...,np.newaxis]))
    V = (R[:, 1][...,np.newaxis], R[:, 0][...,np.newaxis])
    R2 = np.concatenate(V, axis=1)
    R2 = R2.flatten()
    Rs.append(R2)
    Es.append(E/1000)

bin_div = ([0, 5, 10, 20, 30, 40, 50, 60])
bin_indices = []

for model_i in range(len(Rs)):
    bin_indices.append(np.digitize(Es[model_i], bins=bin_div))

curves_mean = np.zeros((np.alen(bin_div), len(Rs)))
curves_variance = np.zeros((np.alen(bin_div), len(Rs)))

for bin_div_i in range(len(bin_div)):
    for model_i in range(len(Rs)):
        RRs = Rs[model_i]
        print(np.shape(RRs))
        RRs = RRs[np.where(bin_indices[model_i] == bin_div_i)]

        print(np.shape(RRs))

        if len(RRs)!=0:
            curves_mean[bin_div_i, model_i] = np.mean(RRs)
            curves_variance[bin_div_i, model_i] = np.var(RRs)
        # else:
        #     curves_mean[bin_div_i, model_i] = -1
        #     curves_variance[bin_div_i, model_i] = -1


plt.gcf()
fig=plt.figure()
fig.set_size_inches(10, 7)


# for i in range(curves_mean.shape[1]):
#     plt.plot(bin_div[1:], curves_mean[1:, i], marker='o', color=colors_2[i])

plt.plot(bin_div[1:], curves_mean[1:, 1], marker='o', color=colors_2[1])
plt.plot(bin_div[1:], curves_mean[1:, 3], marker='o', color=colors_2[3])
plt.plot(bin_div[1:], curves_mean[1:, 2], marker='o', color=colors_2[2])
plt.plot(bin_div[1:], curves_mean[1:, 0], marker='o', color=colors_2[0])

plt.xlabel('Noise shower energy (GeV)')
plt.ylabel('Response (mean)')
plt.legend([models_names[1], models_names[3], models_names[2], models_names[0]])
plt.savefig(os.path.join(args.output, 'mean_response_curve.pdf'))


plt.gcf()
fig=plt.figure()
fig.set_size_inches(10, 7)

for i in range(curves_variance.shape[1]):
    plt.plot(bin_div[1:], curves_variance[1:, i], marker='o')
plt.xlabel('Noise shower energy (GeV)')
plt.ylabel('Response (variance)')
plt.legend(models_names)
plt.savefig(os.path.join(args.output, 'variance_response_curve.pdf'))


print("Plots generated in ", args.output)
