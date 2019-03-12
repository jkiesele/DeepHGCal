import os
import sys
from queue import Queue  # TODO: Write check for python3 (module name is queue in python3)
import argparse
from threading import Thread
import time
import numpy as np
import sys
import tensorflow as tf
import time
from sklearn.utils import shuffle
import sparse_hgcal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.backends.backend_pdf
import random
import matplotlib.colors



max_gpu_events = 500

parser = argparse.ArgumentParser(description='Run multi-process conversion from root to tfrecords')
parser.add_argument('input',
                    help="Path to a sample root file")
# parser.add_argument('output', help="Path where to produce output files")
parser.add_argument("--isGamma", default=False, help="Whether only gamma", type=bool)
args = parser.parse_args()


def make_fixed_array(a,expand=True):
    if expand:
        return np.expand_dims(np.array(a.tolist(), dtype='float32'), axis=2)
    else:
        return np.array(a.tolist(), dtype='float32')

def concat_all_branches(nparray,branches):
    allarrays=[]
    for b in branches:
        allarrays.append(make_fixed_array(nparray[b]))
    return np.concatenate(allarrays,axis=-1)

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def plot_calo(spatial, sizes, colors):
    min_s, max_s = np.min(spatial, axis=0), np.max(spatial, axis=0)
    # map = np.arange(25)
    # np.random.shuffle(map)
    # for i in range(25):
    #     colors[colors==i]=map[i]
    # colors[colors==0]=0
    # colors[colors==1]=3
    # colors[colors==2]=6
    # colors[colors==3]=9
    # colors[colors==4]=12
    # colors[colors==5]=15
    # colors[colors==6]=18
    # colors[colors==8]=21
    # colors[colors==9]=24
    # colors[colors==10]=1
    # colors[colors==11]=4
    # colors[colors==12]=7
    # colors[colors==13]=10
    # colors[colors==14]=13
    # colors[colors==15]=16
    # colors[colors==16]=19
    # colors[colors==17]=22
    # colors[colors==18]=2
    # colors[colors==19]=5
    # colors[colors==20]=8
    # colors[colors==21]=11
    # colors[colors==22]=14
    # colors[colors==23]=17
    # colors[colors==24]=20

    print("Hello", max(colors))

    colorss = ["red", "orange", "gold", "limegreen"]*5
    # colorss = ["red", "blue"] * 10

    cmap = matplotlib.colors.ListedColormap(colorss)

    fig = plt.figure()
    ax = Axes3D(fig)
    im = ax.scatter3D(spatial[:,2],spatial[:,0],spatial[:,1], s=sizes/3, cmap=cmap, c=colors, depthshade=0, edgecolor='')
    # fig.colorbar(im)
    ax.set_xbound(min_s[2], max_s[2])
    ax.set_ybound(min_s[0], max_s[0])
    ax.set_zbound(min_s[1], max_s[1])
    ax.view_init(elev=56, azim=-71)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')

    # plt.show()

    pdf = matplotlib.backends.backend_pdf.PdfPages('/eos/home-s/sqasim/work_pdfs_plots/calo.pdf')
    for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()


def run_conversion_simple(input_file):
    np.set_printoptions(threshold=np.nan)

    location = 'B4'
    branches = ['rechit_x', 'rechit_y', 'rechit_z', 'rechit_vxy', 'rechit_vz', 'rechit_energy', 'rechit_layer',
                'true_x', 'true_y', 'true_r', 'true_energy', 'isGamma']

    types = ['float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64',
             'float64', 'float64', 'int32']

    max_size = [2679 for _ in range(7)] + [1, 1, 1, 1, 1]

    nparray, sizes = sparse_hgcal.read_np_array(input_file, location, branches, types, max_size)

    if args.isGamma:
        isGamma = np.where(nparray[11] == 1)
        nparray = [x[isGamma] for x in nparray]

    true_values_1 = np.concatenate([nparray[i][..., np.newaxis] for i in [7, 8, 9, 10]], axis=1)

    # common part:
    common = concat_all_branches(nparray,
                                 [0, 1, 2, 3, 4, 6])

    positions = concat_all_branches(nparray, [0, 1, 2])[0]
    sizes_1 = make_fixed_array(nparray[3], expand=False)[0]
    colors = make_fixed_array(nparray[6], expand=False)[0]
    sizes_2 = make_fixed_array(nparray[4], expand=False)[0]

    plot_calo(positions, sizes_1, colors)



run_conversion_simple(args.input)
