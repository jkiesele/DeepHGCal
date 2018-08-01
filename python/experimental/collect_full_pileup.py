import numpy as np
import root_numpy
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import numpy_indexed as npi



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

out_pus = []

def work(A, i):
    global out_pus

    X = A['rechit_x'][i]
    Y = A['rechit_y'][i]
    Z = A['rechit_z'][i]
    E = A['rechit_energy'][i]
    L = A['rechit_layer'][i]
    T = A['rechit_time'][i]
    D = A['rechit_detid'][i]

    H = A['simcluster_hits'][i]
    F = A['simcluster_fractions'][i]

    if not(len(X) == len(Y) == len(Z) == len(E) == len(L) == len(T)) or len(X) == 0:
        print("Error in number of entries")
        return

    num_entries = len(X)


    print("Hello, world!")
    # All features
    all_features = np.concatenate((np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1), np.expand_dims(Z, axis=1),
                                   np.expand_dims(E, axis=1), np.expand_dims(L, axis=1), np.expand_dims(T, axis=1)), axis=1)
    total_fractions = np.zeros(len(all_features))

    for k in range(len(H)):
        my_simcluster_hits = H[k]
        my_simcluster_frac = F[k]
        print('\tCluster %d - %d' % (k, len(my_simcluster_hits)))
        valid_indices = np.isin(my_simcluster_hits, D)

        cluster_frac = my_simcluster_frac[valid_indices]
        cluster_hits = my_simcluster_hits[valid_indices]

        cluster_hits = npi.indices(D, cluster_hits, missing='raise')
        total_fractions[cluster_hits] += cluster_frac

    print("Hello, world 2!")

    pu_indices = np.argwhere()[total_fractions < 0.1]
    data_required = A[pu_indices]
    if len(pu_indices) > 300:
        out_pus.append(data_required)

    print("Length", len(out_pus))

    if len(out_pus) == 5:
        return True
    else:
        return False



def convert(input_file, output_file_prefix):
    branches = []
    branches.append('rechit_x')
    branches.append('rechit_y')
    branches.append('rechit_z')
    branches.append('rechit_detid')
    branches.append('rechit_layer')
    branches.append('rechit_energy')
    branches.append('rechit_time')
    branches.append('simcluster_hits')
    branches.append('simcluster_fractions')

    # Initiating the writer and creating the tfrecords file.

    max_entries = 2000

    n_entires = len(root_numpy.root2array(input_file, branches=['event'], treename="ana/hgc")['event'])
    print("Events", n_entires)

    i_entry = 0
    j_entry = 0
    chunk_size = 10
    while j_entry < n_entires:
        A = root_numpy.root2array(input_file, branches=branches, treename="ana/hgc", start=i_entry,
                                  stop=min(i_entry + chunk_size, n_entires))
        for i in range(len(A)):
            done = work(A, i)

            if done:
                break
            j_entry += 1

        if done:
            break

        i_entry += chunk_size

    print("Collected", len(out_pus))
    print("Writing now")

    index = 0
    for i in out_pus:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_pu.bin", 'wb') as f:
            i.tofile(f)
        index += 1

s = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_PU200_20170914/NTUP2/partGun_PDGid11_x160_Pt2.0To100.0_NTUP_190.root'
o = '/afs/cern.ch/work/s/sqasim/workspace_3/FullPU/pu'

convert(s, o)