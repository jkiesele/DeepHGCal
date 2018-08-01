import numpy as np
import os
import sys
import argparse
import root_numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



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
    reduce_function = lambda average, x, axis : np.average(x, axis=axis) if average else np.sum(x, axis=axis)
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


def output_histogram(hist, prefix, xtitle, logscale=False):
    plt.clf()

    if logscale:
        hist = np.log10(hist)

    bin_edges = xbins
    low, high = np.min(hist), np.max(hist)
    plt.plot(bin_edges[:-1], hist, '--', linewidth=0.9)
    # plt.xlim(min(bin_edges), max(bin_edges))
    # plt.ylim(0, max(hist))
    if logscale:
        plt.ylabel('number (log scale)')

    plt.xlabel('Error')
    plt.savefig('%s.png' % prefix)

def output_histogram_2(hist, prefix, xtitle, logscale=False):
    plt.clf()

    if logscale:
        hist = np.log10(hist)

    plt.hist(hist, bins=60)
    # plt.xlim(min(bin_edges), max(bin_edges))
    # plt.ylim(0, max(hist))
    if logscale:
        plt.ylabel('number (log scale)')

    plt.xlabel('Error')
    plt.savefig('%s.png' % prefix)





branches = ['isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy', 'rechit_total_fraction']

xbins = np.linspace(-1,1,400)
error_per_rechit_histogram, _ = np.histogram([],xbins)
per_rechit_fraction_histogram_truth, _ = np.histogram([],xbins)
per_rechit_fraction_histogram_prediction, _ = np.histogram([],xbins)

less_0_1, _ = np.histogram([],xbins)
less_0_5, _ = np.histogram([],xbins)
less_1_0, _ = np.histogram([],xbins)
less_5_0, _ = np.histogram([], xbins)
more_5_0, _ = np.histogram([],xbins)

divided_histos = [less_0_1, less_0_5, less_1_0, less_5_0, more_5_0]


shower_error_percentage = list()
def compare(source, truth, prediction):
    global error_per_rechit_histogram
    global per_rechit_fraction_histogram_truth
    global per_rechit_fraction_histogram_prediction
    global heat_map_diff_fraction
    global shower_error_percentage

    # Pick only energies
    source = source[:, :, :, 0:24:4]

    source_points_indices = np.argwhere(source>float(args.threshold))


    source_flattened = source[source_points_indices[:,0],source_points_indices[:,1],source_points_indices[:,2],
                            source_points_indices[:,3]]

    error = (truth - prediction)
    error_flattened = error[source_points_indices[:,0],source_points_indices[:,1],source_points_indices[:,2],
                            source_points_indices[:,3]]
    new_hist, _ = np.histogram(error_flattened, xbins)
    error_per_rechit_histogram += new_hist

    fraction_truth_flattened = truth[source_points_indices[:,0],source_points_indices[:,1],source_points_indices[:,2],
                            source_points_indices[:,3]]
    per_rechit_fraction_histogram_truth += np.histogram(fraction_truth_flattened, xbins)[0]

    fraction_predict_flattened = prediction[source_points_indices[:,0],source_points_indices[:,1],source_points_indices[:,2],
                            source_points_indices[:,3]]
    per_rechit_fraction_histogram_prediction += np.histogram(fraction_predict_flattened, xbins)[0]

    heat_map_diff_fraction += np.sum(np.abs((truth - prediction)*source), axis=-1)

    shower_error_percentage.append(np.sum(np.abs(error_flattened*source_flattened))/np.sum(source_flattened))


for file_path_pair in file_paths:
    source, prediction = file_path_pair.split(' ')
    A = root_numpy.root2array(prediction+'.y', treename='source/tree')
    B = root_numpy.root2array(prediction+'.yy', treename='prediction/tree')
    C = root_numpy.root2array(prediction+'.x', treename='source/tree')
    D = root_numpy.root2array(source, branches=branches, treename='deepntuplizer/tree')

    nevents = len(A)
    print("Hello, world!", source, nevents, len(B), len(C), len(D))
    for i in range(nevents):
        ii = int(i/2)
        text = find_text(D['isGamma'][ii], D['isElectron'][ii], D['isMuon'][ii], D['isPionCharged'][ii],
                       str(float(D['true_energy'][ii])))
        prediction = np.squeeze(np.reshape(B[i]['data'], B[i]['shape'])).astype(np.float)
        truth = np.squeeze(np.reshape(A[i]['data'], A[i]['shape'])).astype(np.float)

        if len(prediction.shape) == 4:
            source_full_res = np.squeeze(np.reshape(C[i]['data'], C[i]['shape'])).astype(np.float)
            prediction_full_res = np.copy(prediction)
            truth_full_res = np.copy(truth)

            prediction = np.sum(prediction, axis=-1)
            truth = np.sum(truth, axis=-1)
            compare(source_full_res, truth_full_res, prediction_full_res)

        all = np.squeeze(np.reshape(C[i]['data'], C[i]['shape'])).astype(np.float)
        if args.event:
            plot_fractions(truth, prediction, all, text, float(D['true_energy'][ii]))


        heat_map_total_fraction_source += truth
        heat_map_total_fraction_prediction += prediction
        # heat_map_diff_fraction += np.abs(truth-prediction)
        heat_map_ratio_fraction += truth/prediction
        heat_map_diff_fraction_squared += np.square(truth-prediction)

    total_events += nevents
    # if nevents >= 7500:
    #     break
    break

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

output_histogram(error_per_rechit_histogram, args.output+'/error_per_rechit_histo', xtitle='Error')
output_histogram(per_rechit_fraction_histogram_truth, args.output+'/per_rechit_fraction_truth_histo', xtitle='Fraction truth', logscale=True)
output_histogram(per_rechit_fraction_histogram_prediction, args.output+'/per_rechit_fraction_predict_histo', xtitle='Fraction prediction', logscale=True)
output_histogram_2(shower_error_percentage, args.output+'/shower_error_percentage', xtitle='Error')


