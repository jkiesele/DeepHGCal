import numpy as np
import numpy as np
import ROOT
import root_numpy
from root_numpy import tree2array, root2array
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
import random
import sys
import timeit


branches = []
branches.append('simcluster_eta')
branches.append('rechit_x')
branches.append('rechit_y')
branches.append('rechit_z')
branches.append('rechit_detid')
branches.append('rechit_energy')
branches.append('simcluster_hits')
branches.append('simcluster_fractions')
branches.append('genpart_ovx')
branches.append('genpart_ovy')
branches.append('genpart_ovz')
branches.append('genpart_dvx')
branches.append('genpart_dvy')
branches.append('genpart_dvz')
branches.append('genpart_exx')
branches.append('genpart_exy')
branches.append('genpart_eta')
branches.append('genpart_phi')


def plot_rechits(x, y, z, cs, c, ax):
    ax.scatter(z, x, y, c=c, s=np.log(cs+1)*100, cmap=cmx.hsv)


def plot_particles(A, index,ax, only_calorimeter_particles):
    ovx = A['genpart_ovx'][index]
    ovy = A['genpart_ovy'][index]
    ovz = A['genpart_ovz'][index]
    dvz = A['genpart_dvz'][index]

    dvx = (np.abs(dvz) < 320)*A['genpart_dvx'][index]+(np.abs(dvz) >= 320)*A['genpart_exx'][index]
    dvy = (np.abs(dvz) < 320)*A['genpart_dvy'][index] + (np.abs(dvz) >= 320)*A['genpart_exy'][index]

    if only_calorimeter_particles:
        interesting_indexes = np.where((np.abs(ovz) <= 0.001) * (np.abs(dvz) > 0.001))
        ovx = ovx[interesting_indexes]
        ovy = ovy[interesting_indexes]
        ovz = ovz[interesting_indexes]
        dvx = dvx[interesting_indexes]
        dvy = dvy[interesting_indexes]
        dvz = dvz[interesting_indexes]

    num_particles = int(np.size(ovx))
    print("Calorimeter particles", num_particles)
    for i in range(num_particles):
        xs = np.linspace(ovx[i], dvx[i])
        ys = np.linspace(ovy[i], dvy[i])
        zs = np.linspace(ovz[i], dvz[i])

        ax.plot(zs, xs, ys)


if __name__ =="__main__":
    if len(sys.argv) < 2:
        print("Usage: python bin/plot_events path/to/root_file.root")
        exit()

    file = sys.argv[1]
    A = root_numpy.root2array(file, branches=branches)
    plot_only_calorimeter_particles = False
    if len(sys.argv) >= 3:
        plot_only_calorimeter_particles = sys.argv[2] == 'only_calorimeter_particles'
    if plot_only_calorimeter_particles:
        print("Plotting only calorimeter particles")

    for i_entry in range(192):
        num_simclusters = int(np.size(A['simcluster_hits'][i_entry]))

        X=np.array([])
        Y=np.array([])
        Z=np.array([])
        E=np.array([])
        C=np.array([])

        for j_simcluster in range(num_simclusters):
            start = timeit.timeit()
            hits_simcluster = A['simcluster_hits'][i_entry][j_simcluster]
            indices = np.where(np.in1d(A['rechit_detid'][i_entry], hits_simcluster))[0]
            # print("Hello, world!")
            end = timeit.timeit()
            print end - start


            hits_simcluster = A['simcluster_hits'][i_entry][j_simcluster]
            indices = np.where(np.in1d(A['rechit_detid'][i_entry], hits_simcluster))[0]

            x = A['rechit_x'][i_entry][indices]
            y = A['rechit_y'][i_entry][indices]
            z = A['rechit_z'][i_entry][indices]
            e = np.log(A['rechit_energy'][i_entry][indices]+1)

            X = np.concatenate((X,x),axis=0)
            Y = np.concatenate((Y,y),axis=0)
            Z = np.concatenate((Z,z),axis=0)
            E = np.concatenate((E,e),axis=0)
            C = np.concatenate((C,np.ones_like(x) * j_simcluster),axis=0)

        fig = plt.figure(1)
        ax = Axes3D(fig)
        plot_rechits(X, Y, Z, E, C, ax)
        plot_particles(A, i_entry,ax, plot_only_calorimeter_particles)
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        plt.show()