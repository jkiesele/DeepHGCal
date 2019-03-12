import matplotlib as mpl
# mpl.use('TkAgg')
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.backends.backend_pdf


def plot_clustering_4(spatial, energy, prediction ,gt, do_plot=True, elev=None, azmuth=None):
    min_s, max_s = np.min(spatial, axis=0), np.max(spatial, axis=0)


    prediction_0 = prediction[:,0]
    prediction_1 = prediction[:,1]

    gt_0 = gt[:,0]
    gt_1 = gt[:,1]

    #
    # fig = plt.figure(1)
    # ax = Axes3D(fig)
    # jet = plt.get_cmap('PiYG')
    # ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy*gt_0+0.1)*5, c='red', cmap=jet)
    # ax.set_xbound(min_s[2], max_s[2])
    # ax.set_ybound(min_s[0], max_s[0])
    # ax.set_zbound(min_s[1], max_s[1])
    # ax.set_xlabel('z (mm)')
    # ax.set_ylabel('x (mm)')
    # ax.set_zlabel('y (mm)')
    #
    # fig = plt.figure(2)
    # ax = Axes3D(fig)
    # jet = plt.get_cmap('PiYG')
    # ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy*gt_1+0.1)*5, c='red', cmap=jet)
    # ax.set_xbound(min_s[2], max_s[2])
    # ax.set_ybound(min_s[0], max_s[0])
    # ax.set_zbound(min_s[1], max_s[1])
    # ax.set_xlabel('z (mm)')
    # ax.set_ylabel('x (mm)')
    # ax.set_zlabel('y (mm)')
    #
    # fig = plt.figure(3)
    # ax = Axes3D(fig)
    # jet = plt.get_cmap('PiYG')
    # ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy*prediction_0+0.1)*5, c='green', cmap=jet)
    # ax.set_xbound(min_s[2], max_s[2])
    # ax.set_ybound(min_s[0], max_s[0])
    # ax.set_zbound(min_s[1], max_s[1])
    #
    # fig = plt.figure(4)
    # ax = Axes3D(fig)
    # jet = plt.get_cmap('PiYG')
    # ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy*prediction_1+0.1)*5, c='green', cmap=jet)
    # ax.set_xbound(min_s[2], max_s[2])
    # ax.set_ybound(min_s[0], max_s[0])
    # ax.set_zbound(min_s[1], max_s[1])
    #
    # fig = plt.figure(5)
    # ax = Axes3D(fig)
    # jet = plt.get_cmap('PiYG')
    # ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy)*5, c='orange', cmap=jet)
    # ax.set_xbound(min_s[2], max_s[2])
    # ax.set_ybound(min_s[0], max_s[0])
    # ax.set_zbound(min_s[1], max_s[1])


    fig = plt.figure()
    # fig.set_size_inches(10, 7)
    ax = Axes3D(fig)
    if elev is not None:
        ax.view_init(elev, azmuth)
    jet = plt.get_cmap('seismic')
    ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.sqrt(energy)*2, c=prediction_0, cmap=jet)
    ax.set_xbound(min_s[2], max_s[2])
    ax.set_ybound(min_s[0], max_s[0])
    ax.set_zbound(min_s[1], max_s[1])
    fig.canvas.set_window_title('Pred')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')


    fig = plt.figure()
    # fig.set_size_inches(10, 7)
    ax = Axes3D(fig)
    if elev is not None:
        ax.view_init(elev, azmuth)
    jet = plt.get_cmap('seismic')
    ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.sqrt(energy)*2, c=gt_0, cmap=jet)
    ax.set_xbound(min_s[2], max_s[2])
    ax.set_ybound(min_s[0], max_s[0])
    ax.set_zbound(min_s[1], max_s[1])
    fig.canvas.set_window_title('GT')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')


    if do_plot:
        plt.show()


def plot_clustering_41(spatial, energy, prediction ,gt):
    min_s, max_s = np.min(spatial, axis=0), np.max(spatial, axis=0)


    prediction_0 = prediction[:,0]
    prediction_1 = prediction[:,1]

    gt_0 = gt[:,0]
    gt_1 = gt[:,1]


    fig = plt.figure(1)
    ax = Axes3D(fig)
    jet = plt.get_cmap('PiYG')
    ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy+0.1)*5, c=prediction_0, cmap=jet)
    ax.set_xbound(min_s[2], max_s[2])
    ax.set_ybound(min_s[0], max_s[0])
    ax.set_zbound(min_s[1], max_s[1])
    fig.canvas.set_window_title('Pred')

    fig = plt.figure(3)
    ax = Axes3D(fig)
    jet = plt.get_cmap('PiYG')
    ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy+0.1)*5, c=gt_0, cmap=jet)
    ax.set_xbound(min_s[2], max_s[2])
    ax.set_ybound(min_s[0], max_s[0])
    ax.set_zbound(min_s[1], max_s[1])
    fig.canvas.set_window_title('GT')

    # fig = plt.figure(5)
    # ax = Axes3D(fig)
    # jet = plt.get_cmap('PiYG')
    # ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.log(energy)*5, c='orange', cmap=jet)
    # ax.set_xbound(min_s[2], max_s[2])
    # ax.set_ybound(min_s[0], max_s[0])
    # ax.set_zbound(min_s[1], max_s[1])



    plt.show()




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



def plot_clustering_layer_wise_visualize(spatial, energy, prediction ,gt, layer_feats, config_name):
    print("Hello, good people!")
    print(spatial.shape)
    print(energy.shape)
    print(prediction.shape)
    print(gt.shape)
    print([x.shape for x in layer_feats])
    print(type(spatial))

    print(type(spatial), type(energy), type(prediction), type(gt))

    min_s, max_s = np.min(spatial, axis=0), np.max(spatial, axis=0)

    prediction_0 = prediction[:, 0]
    prediction_1 = prediction[:, 1]

    gt_0 = gt[:, 0]
    gt_1 = gt[:, 1]

    i = np.where((spatial[:, 0] < 50) & (spatial[:, 0] > 0) & (spatial[:, 1] > 0) & (spatial[:, 1]<50) & (energy > 30) & (gt_0<0.5))
    print(i)
    # vindex = 1094#1985#i[0][33]
    # vindex = 1500
    vindex = 719
    print(energy[i])
    print(spatial[i])
    print(prediction_0[i])


    plot_clustering_4(spatial, energy, prediction, gt, do_plot=False, elev=56, azmuth=-71)

    for i in range(len(layer_feats)):
        print("What the fuck is this?")
        layer_feat_mine = layer_feats[i][vindex]
        # distances = np.exp(-np.sum((layer_feats[i] - layer_feat_mine[np.newaxis, ...])**2, axis=-1))
        if 'dgcnn' in config_name:
            distances = np.sum((layer_feats[i] - layer_feat_mine[np.newaxis, ...])**2, axis=-1)
        else:
            distances = np.exp(-np.sum((layer_feats[i] - layer_feat_mine[np.newaxis, ...])**2, axis=-1))

        fig = plt.figure()
        # fig.set_size_inches(10, 7)
        ax = Axes3D(fig)
        ax.view_init(elev=56, azim=-71)
        jet = plt.get_cmap('Oranges')
        # plt.title("Layer: %d" % i)
        ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.sqrt(energy)*2, c=distances, cmap=jet)
        ax.scatter(spatial[vindex,2],spatial[vindex,0],spatial[vindex,1], s=1000, c='green', cmap=jet)
        ax.set_xbound(min_s[2], max_s[2])
        ax.set_ybound(min_s[0], max_s[0])
        ax.set_zbound(min_s[1], max_s[1])
        fig.canvas.set_window_title('Dist '+str(i))
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('x (mm)')
        ax.set_zlabel('y (mm)')


    output_file_name = '/afs/cern.ch/user/s/sqasim/'+config_name+'.pdf'

    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file_name)
    for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()
    0/0

def plot_clustering_layer_wise_visualize_agg(spatial, energy, prediction ,gt, layer_feats, config_name):
    print("Hello, good people!")
    print(spatial.shape)
    print(energy.shape)
    print(prediction.shape)
    print(gt.shape)
    print([x.shape for x in layer_feats])
    print(type(spatial))

    print(type(spatial), type(energy), type(prediction), type(gt))

    min_s, max_s = np.min(spatial, axis=0), np.max(spatial, axis=0)

    prediction_0 = prediction[:, 0]
    prediction_1 = prediction[:, 1]


    i = np.where((spatial[:, 0] < 50) & (spatial[:, 0] > 0) & (spatial[:, 1] > 0) & (spatial[:, 1]<50) & (energy > 30) & (prediction_0<0.5))
    print(i)
    vindex = 1094#1985#i[0][33]
    print(energy[i])
    print(spatial[i])
    print(prediction_0[i])


    gt_0 = gt[:, 0]
    gt_1 = gt[:, 1]

    plot_clustering_4(spatial, energy, prediction, gt, do_plot=False, elev=56, azmuth=-71)

    for i in range(len(layer_feats)):
        layer_feat_mine = layer_feats[i][vindex]
        # distances = np.exp(-np.sum((layer_feats[i] - layer_feat_mine[np.newaxis, ...])**2, axis=-1))
        # F = layer_feat_mine.shape[1]
        for j in range(4):
            distances = np.exp(-np.abs((layer_feats[i])[:,j]))
            print(distances.shape)

            fig = plt.figure()
            # fig.set_size_inches(10, 7)
            ax = Axes3D(fig)
            # plt.title("Layer: %d - Coordinate: %d" % (i+1,j+1))
            ax.view_init(elev=56, azim=-71)
            jet = plt.get_cmap('Oranges')
            ax.scatter(spatial[:,2],spatial[:,0],spatial[:,1], s=np.sqrt(energy)*2, c=distances, cmap=jet)
            # ax.scatter(spatial[vindex,2],spatial[vindex,0],spatial[vindex,1], s=1000, c='green', cmap=jet)
            ax.set_xbound(min_s[2], max_s[2])
            ax.set_ybound(min_s[0], max_s[0])
            ax.set_zbound(min_s[1], max_s[1])
            fig.canvas.set_window_title('Dist '+str(i))
            ax.set_xlabel('z (mm)')
            ax.set_ylabel('x (mm)')
            ax.set_zlabel('y (mm)')

    output_file_name = '/eos/home-s/sqasim/work_pdfs_plots/'+config_name+'.pdf'

    pdf = matplotlib.backends.backend_pdf.PdfPages(output_file_name)
    for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()
    0/0

    # plt.show()


