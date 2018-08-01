import numpy as np
import root_numpy
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D


def euclidean_two_datasets(A, B):
    """
    Returns euclidean distance between two datasets

    A is first dataset in form (N,F) where N is number of examples in first dataset, F is number of features
    B is second dataset in form (M,F) where M is number of examples in second dataset, F is number of features

    Returns:
    A matrix of size (N,M) where each element (i,j) denotes euclidean distance between ith entry in first dataset and jth in second dataset

    """
    A = np.array(A)
    B = np.array(B)
    return np.sqrt(-2*A.dot(B.transpose()) + (np.sum(B*B,axis=1)) + (np.sum(A*A,axis=1))[:,np.newaxis])


out_pion_charged = list()
out_pion_charged_without_pu = list()
out_pion_charged_only_pu = list()

out_photon = list()
out_photon_without_pu = list()
out_photon_only_pu = list()




def plot_rechits(x,y, z, energy, text):
    fig = plt.figure(0)
    ax = Axes3D(fig)

    x_min = min(0,np.min(x))
    y_min = min(0,np.min(y))
    z_min = min(0,np.min(z))

    x_max = max(0,np.max(x))
    y_max = max(0,np.max(y))
    z_max = max(0,np.max(z))

    xx, yy = np.meshgrid(np.linspace(x_min,x_max), np.linspace(y_min,y_max))
    zz = 320 * np.ones(np.shape(xx))
    ax.plot_surface(xx, yy, zz, alpha=0.3,cmap=plt.cm.RdYlBu_r)
    ax.plot_surface(xx, yy, -zz, alpha=0.3,cmap=plt.cm.RdYlBu_r)
    ax.scatter(x, y, z, s=np.log(energy+1)*100,  cmap=cmx.hsv)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title("Everything "+text)

    ax.set_xbound(x_min, x_max)
    ax.set_ybound(y_min, y_max)
    ax.set_zbound(z_min, z_max)

    ax.set_title(text)


index = 0
def work(A, i, output_file_prefix):
    print("Hello world", index)
    global index
    X = A['rechit_x'][i]
    Y = A['rechit_y'][i]
    Z = A['rechit_z'][i]
    E = A['rechit_energy'][i]
    L = A['rechit_layer'][i]
    T = A['rechit_time'][i]

    ex = lambda X: np.expand_dims(X, axis=1)
    X = ex(X)
    Y = ex(Y)
    Z = ex(Z)
    E = ex(E)
    L = ex(L)
    T = ex(T)

    print(X.dtype, Y.dtype, Z.dtype, E.dtype, L.dtype, T.dtype)


    if not(len(X) == len(Y) == len(Z) == len(E) == len(L) == len(T)) or len(X) == 0:
        print("Error in number of entries")
        return

    J = np.concatenate((X, Y, Z, E, L, T), axis=1)

    print(J.dtype)

    print(J)


    0/0

    with open(output_file_prefix + "_" + str(index) + ".bin2", 'wb') as f:
        J.tofile(f)

    index += 1


def convert(input_file, output_file_prefix):
    branches = []
    branches.append('rechit_x')
    branches.append('rechit_y')
    branches.append('rechit_z')
    branches.append('rechit_layer')
    branches.append('rechit_energy')
    branches.append('rechit_time')

    # Initiating the writer and creating the tfrecords file.

    max_entries = 2000

    print(input_file)

    n_entires = 100
    print("Events", n_entires)

    i_entry = 0
    j_entry = 0
    chunk_size = 200
    while j_entry < n_entires:
        A = root_numpy.root2array(input_file, branches=branches, treename="ana/hgc", start=i_entry,
                                  stop=min(i_entry + chunk_size, n_entires))
        for i in range(len(A)):
            done = work(A, i, output_file_prefix)

            if done:
                break
            j_entry += 1

        if done:
            break

        i_entry += chunk_size


input_file = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/RelValTTbar_14TeV_CMSSW_10_0_0_pre1-PU25ns_94X_upgrade2023_realistic_v2_2023D17PU200-v1_GEN-SIM-RECO_NTUP/_RelValTTbar_14TeV_CMSSW_10_0_0_pre1-PU25ns_94X_upgrade2023_realistic_v2_2023D17PU200-v1_GEN-SIM-RECO_NTUP_10.root'
output_file_prefix = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/RelValTTbar_14TeV_CMSSW_10_0_0_pre1-PU25ns_94X_upgrade2023_realistic_v2_2023D17PU200-v1_GEN-SIM-RECO_NTUP/converted/bla_bla'

convert(input_file, output_file_prefix)