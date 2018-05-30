import numpy as np
import sparse_hgcal as hg
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import matplotlib.cm as cmx


def find_type(A):
    assert np.size(A) == 6

    if A[0] == 1:
        return "Electron"
    elif A[1] == 1:
        return "Muon"
    elif A[2] == 1:
        return "Charged Pion"
    elif A[3] == 1:
        return "Neutral Pion"
    elif A[4] == 1:
        return "K0 Long"
    elif A[5] == 1:
        return "K0 Short"


parser = argparse.ArgumentParser(description='Plot denoising output')
parser.add_argument('input',
                    help="Path to the file which you want to plot")
args = parser.parse_args()

file = args.input
location = 'B4'
branches = ['isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short', 'rechit_energy',
            'rechit_x', 'rechit_y', 'rechit_z']
types = ['int32', 'int32', 'int32', 'int32', 'int32', 'int32', 'float64', 'float64', 'float64', 'float64']
max_sizes = [1, 1, 1, 1, 1, 1, 3000, 3000, 3000, 3000]

print("Loading data")
data, sizes = hg.read_np_array(file, location, branches, types, max_sizes)
print("Data loaded")

events = np.size(data[0])

E = data[6]
X = data[7]
Y = data[8]
Z = data[9]

T = np.concatenate((np.expand_dims(data[0], 1), np.expand_dims(data[1], 1), np.expand_dims(data[2], 1),
                    np.expand_dims(data[3], 1), np.expand_dims(data[4], 1), np.expand_dims(data[5], 1)), axis=1)


def plot_rechits(X, Y, Z, E, text):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(Z, X, Y, s=np.log(E + 1) * 100, cmap=cmx.hsv)
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    ax.set_title(text)
    plt.show()

for i in range(events):
    s = sizes[6][i]
    print("Hits", s)
    XX = X[i][0:s]
    YY = Y[i][0:s]
    ZZ = Z[i][0:s]
    EE = E[i][0:s]
    TT = T[i]
    text = find_type(TT)

    plot_rechits(XX, YY, ZZ, EE, text)
