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


def work(A, i, particle, pileup):
    global out_pion_charged
    global out_pion_charged_without_pu
    global out_pion_charged_only_pu

    global out_photon
    global out_photon_without_pu
    global out_photon_only_pu


    X = A['rechit_x'][i]
    Y = A['rechit_y'][i]
    Z = A['rechit_z'][i]
    E = A['rechit_energy'][i]
    L = A['rechit_layer'][i]
    T = A['rechit_time'][i]
    F = A['rechit_total_fraction'][i]

    if not(len(X) == len(Y) == len(Z) == len(E) == len(L) == len(T)) or len(X) == 0:
        print("Error in number of entries")
        return

    num_entries = len(X)

    # X = X[0:min(num_entries, 2000)]
    # Y = Y[0:min(num_entries, 2000)]
    # Z = Z[0:min(num_entries, 2000)]
    # E = E[0:min(num_entries, 2000)]
    try:
        if len(np.squeeze(np.argwhere(F < 0.1))) > 50 and len(np.squeeze(np.argwhere(F > 0.1))) > 50:
            # All features
            all_features = np.concatenate((np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1), np.expand_dims(Z, axis=1),
                                           np.expand_dims(E, axis=1), np.expand_dims(L, axis=1), np.expand_dims(T, axis=1)), axis=1)

            if int(A['isPionCharged'][i]) == 1 and len(out_pion_charged_without_pu) < 20:
                indices_without_pu = np.squeeze(np.argwhere(F > 0.1))
                indices_only_pu = np.squeeze(np.argwhere(F <= 0.1))

                out_pion_charged.append(all_features)
                out_pion_charged_without_pu.append(all_features[indices_without_pu, :])
                out_pion_charged_only_pu.append(all_features[indices_only_pu, :])

                print(len(all_features), len(indices_only_pu), len(indices_without_pu))

            elif int(A['isGamma'][i]) == 1 and len(out_photon_without_pu) < 20:
                indices_without_pu = np.squeeze(np.argwhere(F > 0.1))
                indices_only_pu = np.squeeze(np.argwhere(F <= 0.1))

                out_photon.append(all_features)
                out_photon_without_pu.append(all_features[indices_without_pu, :])
                out_photon_only_pu.append(all_features[indices_only_pu, :])

                print(len(all_features), len(indices_only_pu), len(indices_without_pu))


            # print(particle, pileup)
            # print(len(X), len(all_features))
            # plot_rechits(all_features[:, 0], all_features[:, 1], all_features[:, 2], all_features[:, 3], "bla")
            # plt.show()
            # out.append(all_features)

    except:
        print("Excepting")

    print("Length", len(out_pion_charged_only_pu), len(out_photon_only_pu))

    if len(out_pion_charged_only_pu) == 20 and len(out_photon_only_pu) == 20:
        return True
    else:
        return False



def convert(input_file, output_file_prefix, particle, pileup):
    branches = []
    branches.append('rechit_x')
    branches.append('rechit_y')
    branches.append('rechit_z')
    branches.append('rechit_layer')
    branches.append('rechit_energy')
    branches.append('rechit_time')
    branches.append('rechit_total_fraction')

    branches.append('isGamma')
    branches.append('isMuon')
    branches.append('isPionCharged')

    # Initiating the writer and creating the tfrecords file.

    max_entries = 2000

    print(input_file)

    n_entires = len(root_numpy.root2array(input_file, branches=['isElectron'], treename="deepntuplizer/tree")['isElectron'])
    print("Events", n_entires)

    i_entry = 0
    j_entry = 0
    chunk_size = 200
    while j_entry < n_entires:
        A = root_numpy.root2array(input_file, branches=branches, treename="deepntuplizer/tree", start=i_entry,
                                  stop=min(i_entry + chunk_size, n_entires))
        for i in range(len(A)):
            done = work(A, i, particle, pileup)

            if done:
                break
            j_entry += 1

        if done:
            break

        i_entry += chunk_size

    print("Collected", len(out_photon_without_pu), len(out_pion_charged_without_pu))
    print("Writing now")

    index = 0
    for i in out_pion_charged:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_pion_charged.bin", 'wb') as f:
            i.tofile(f)
        index += 1

    index = 0
    for i in out_pion_charged_only_pu:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_pion_charged_only_pu.bin", 'wb') as f:
            i.tofile(f)
        index += 1

    index = 0
    for i in out_pion_charged_without_pu:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_pion_charged_without_pu.bin", 'wb') as f:
            i.tofile(f)
        index += 1

    index = 0
    for i in out_photon:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_photon.bin", 'wb') as f:
            i.tofile(f)
        index += 1

    index = 0
    for i in out_photon_only_pu:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_photon_only_pu.bin", 'wb') as f:
            i.tofile(f)
        index += 1

    index = 0
    for i in out_photon_without_pu:
        print("Writing ", index)
        with open(output_file_prefix+"_"+str(index)+"_photon_without_pu.bin", 'wb') as f:
            i.tofile(f)
        index += 1

