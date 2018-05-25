import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import os


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


def show(file):
    with open(file, 'rb') as f:
        all_features = np.fromfile(f, dtype=np.float32)
        all_features = np.reshape(all_features, (-1, 6))
        plot_rechits(all_features[:, 0], all_features[:, 1], all_features[:, 2], all_features[:, 3], os.path.split(file)[1])
        plt.show()


with open(sys.argv[1]) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


for item in file_paths:
    show(item)
