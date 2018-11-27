import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_clustering(spatial, energy, prediction, threshold=0.0001, fig=None):
    min_s, max_s = np.min(spatial, axis=0), np.max(spatial, axis=0)

    rechit_indices = np.where(energy > threshold)

    energy = energy[rechit_indices]
    prediction = prediction[rechit_indices]
    spatial = spatial[rechit_indices]

    cluster = np.argmin(prediction, axis=-1)

    shower_1_indices = np.where(cluster==0)
    shower_2_indices = np.where(cluster==1)

    shower_1_spatial = spatial[shower_1_indices]
    shower_2_spatial = spatial[shower_2_indices]
    shower_1_energy = energy[shower_1_indices]
    shower_2_energy = energy[shower_2_indices]

    shower_1_energy_sizes = np.log(shower_1_energy+0.1)*5
    shower_2_energy_sizes = np.log(shower_2_energy+0.1)*5


    energy_sizes = np.log(energy+0.1)*5


    if fig is None:
        fig = plt.figure()
    ax = Axes3D(fig)
    jet = plt.get_cmap('PiYG')

    ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=energy_sizes, c=prediction[:, 0], cmap=jet)
    ax.set_xbound(min_s[2], max_s[2])
    ax.set_ybound(min_s[0], max_s[0])
    ax.set_zbound(min_s[1], max_s[1])

    if fig is None:
        plt.show()






