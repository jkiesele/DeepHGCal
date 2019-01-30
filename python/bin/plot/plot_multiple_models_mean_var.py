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

models_names = []


for p in os.listdir(args.input):
    if not p.endswith('.pbin'):
        continue
    if 'single_neighbours_plusmean' in p:
        models_names.append('LDSFT')
    elif 'hidden_aggregators_plusmean' in p:
        models_names.append('Aggregator')
    elif 'dgcnn' in p:
        models_names.append('DGCNN')
    elif 'binning_clustering_epsilon_1' in p:
        models_names.append('Binning')
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
    Es.append(E)

bin_div = ([0, 5000, 10000, 20000, 30000, 40000, 50000, 60000])
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


for i in range(curves_mean.shape[1]):
    plt.plot(bin_div[1:], curves_mean[1:, i], marker='o')
plt.xlabel('Noise shower energy (MeV)')
plt.ylabel('Response (mean)')
plt.legend(models_names)
plt.savefig(os.path.join(args.output, 'mean_response_curve.pdf'))

plt.gcf()
fig=plt.figure()
fig.set_size_inches(10, 7)

for i in range(curves_variance.shape[1]):
    plt.plot(bin_div[1:], curves_variance[1:, i], marker='o')
plt.xlabel('Noise shower energy (MeV)')
plt.ylabel('Response (variance)')
plt.legend(models_names)
plt.savefig(os.path.join(args.output, 'variance_response_curve.pdf'))
