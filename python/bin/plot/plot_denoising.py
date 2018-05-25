import numpy as np
import os
import sys
import argparse
import root_numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



parser = argparse.ArgumentParser(description='Plot denoising output')
parser.add_argument('input',
                    help="Path to tree association file")
parser.add_argument('output', help="Path where to produce output files")
parser.add_argument('--event', default=False, help="3d event by event")
parser.add_argument('--threshold', default=0.4, help="3d event by event")
args = parser.parse_args()


def plot_fractions(source, prediction, all, text, true_energy):
    global args
    source_points = np.argwhere(source>float(args.threshold))
    prediction_points = np.argwhere(prediction>float(args.threshold))
    indexes = np.array([3,7,11,15,19,23]).astype(np.int32)
    indexes = np.array([2,6,10,14,18,22]).astype(np.int32)
    all = np.sum(all[:,:,:,indexes], axis=3)
    all_points = np.argwhere(all!=0)
    spec = all[all_points[:,0],all_points[:,1],all_points[:,2]]
    spec_source = all[source_points[:,0],source_points[:,1],source_points[:,2]]
    spec_prediction = all[prediction_points[:,0],prediction_points[:,1],prediction_points[:,2]]

    energy_fractioned_predicted = all*prediction
    energy_fractioned_source = all*source
    print("Predicted: %4.2f Source: %4.2f Truth: %4.2f" % (np.sum(energy_fractioned_predicted), np.sum(energy_fractioned_source), true_energy))

    x_max, y_max, z_max = np.shape(source)

    fig1 = plt.figure(1)
    ax = Axes3D(fig1)
    ax.scatter(source_points[:,0], source_points[:,1], source_points[:,2], s=np.log(spec_source+1)*100)
    ax.set_xbound(0, x_max)
    ax.set_ybound(0, y_max)
    ax.set_zbound(0, z_max)
    plt.title('Source '+text)
    fig2 = plt.figure(2)
    ax = Axes3D(fig2)
    plt.title('Prediction'+text)
    ax.scatter(prediction_points[:,0], prediction_points[:,1], prediction_points[:,2], s=np.log(spec_prediction+1)*100)
    ax.set_xbound(0, x_max)
    ax.set_ybound(0, y_max)
    ax.set_zbound(0, z_max)
    fig2 = plt.figure(3)
    ax = Axes3D(fig2)
    plt.title('All'+text)
    ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], s=np.log(spec+1)*100)
    ax.set_xbound(0, x_max)
    ax.set_ybound(0, y_max)
    ax.set_zbound(0, z_max)
    plt.show()


with open(args.input) as f:
    content = f.readlines()
file_paths = [x.strip() for x in content]


branches = ['rechit_total_fraction']

total_events = 0
heat_map_total_fraction_source = np.zeros((13,13,55))
heat_map_total_fraction_prediction = np.zeros((13,13,55))

heat_map_diff_fraction = np.zeros((13,13,55))
heat_map_diff_fraction_squared = np.zeros((13,13,55))
heat_map_ratio_fraction = np.zeros((13,13,55))


def find_text(isGama, isElectron, isMuon, isPionCharged, trueEnergy):
    s = ""
    if isGama:
        s = "Gamma"
    elif isElectron:
        s = "Electron"
    elif isMuon:
        s = "Muon"
    elif isPionCharged:
        s = "Pion Charged"
    else:
        s = "Unknown"
    return s + " " + trueEnergy + " GeV"

def output_axis_maps(tensor, prefix, average=False):
    global args
    reduce_function = lambda average,x,axis : np.average(x, axis=axis) if average else np.sum(x, axis=axis)
    heat_eta_phi = reduce_function(average, tensor, 2)
    heat_eta_layers = reduce_function(average, tensor, 1)
    heat_phi_layers = reduce_function(average, tensor, 0)

    plt.clf()
    plt.imshow(heat_eta_phi, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('%s_eta_phi.png' % prefix)
    plt.clf()
    plt.imshow(heat_eta_layers, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('%s_eta_layers.png' % prefix)
    plt.clf()
    plt.imshow(heat_phi_layers, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig('%s_phi_layers.png' % prefix)



branches = ['isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy', 'rechit_total_fraction']


for file_path_pair in file_paths:
    source, prediction = file_path_pair.split(' ')
    A = root_numpy.root2array(prediction+'.y', treename='source/tree')
    B = root_numpy.root2array(prediction+'.yy', treename='prediction/tree')
    C = root_numpy.root2array(prediction+'.x', treename='source/tree')
    D = root_numpy.root2array(source, branches=branches, treename='deepntuplizer/tree')

    assert len(A) == len(B)
    nevents = len(A)
    for i in range(nevents):
        text=find_text(D['isGamma'][i], D['isElectron'][i], D['isMuon'][i], D['isPionCharged'][i],
                         str(float(D['true_energy'][i])))
        prediction = np.squeeze(np.reshape(B[i]['data'], B[i]['shape'])).astype(np.float)
        truth = np.squeeze(np.reshape(A[i]['data'], A[i]['shape'])).astype(np.float)
        all = np.squeeze(np.reshape(C[i]['data'], C[i]['shape'])).astype(np.float)
        if args.event:
            plot_fractions(truth, prediction, all, text, float(D['true_energy'][i]))
        heat_map_total_fraction_source += truth
        heat_map_total_fraction_prediction += prediction
        heat_map_diff_fraction += np.abs(truth-prediction)
        heat_map_ratio_fraction += truth/prediction
        heat_map_diff_fraction_squared += np.square(truth-prediction)

    total_events += nevents

print("Total events", total_events)
print("Total source fraction sum", np.sum(heat_map_total_fraction_source))
print("Average source fraction sum", np.sum(heat_map_total_fraction_source) / nevents)
print("Total prediction fraction sum", np.sum(heat_map_total_fraction_prediction))
print("Average prediction fraction sum", np.sum(heat_map_total_fraction_prediction) / nevents)

output_axis_maps(heat_map_total_fraction_source, args.output+'/'+'heat_total_fraction_source')
output_axis_maps(heat_map_total_fraction_prediction, args.output+'/'+'heat_total_fraction_prediction')
output_axis_maps(heat_map_diff_fraction / total_events, args.output+'/'+'heat_total_fraction_diff_abs', average=True)
output_axis_maps(heat_map_diff_fraction_squared / total_events, args.output+'/'+'heat_total_fraction_diff_squared', average=True)
output_axis_maps(heat_map_ratio_fraction, args.output+'/'+'heat_total_fraction_ratio')

