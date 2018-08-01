import root_numpy
from bins.binner import *


branches = ['isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy', 'rechit_time',
            'rechit_energy', 'rechit_phi', 'rechit_eta','rechit_layer', 'seed_eta', 'seed_phi']


f1 = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/convertedH_FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_PU200_20170914/fourth/train_fourth/partGun_PDGid11_x160_Pt2.0To100.0_NTUP_152.root'


A = root_numpy.root2array(f1, treename='deepntuplizer/tree', branches=branches)


num_entries = len(A['seed_eta'])
print("Number of entries is ", num_entries)

for i in range(num_entries):
    # N rechits per event
    rechit_energy = A['rechit_energy'][i] # N energy
    rechit_time = A['rechit_time'][i] # N time
    rechit_eta = A['rechit_eta'][i]
    rechit_phi = A['rechit_phi'][i]
    rechit_layer = A['rechit_layer'][i]

    seed_eta = A['seed_eta'][i] # scalar eta of the seed
    seed_phi = A['seed_phi'][i] # scalar phi of the seed

    pic3d = process(rechit_eta, rechit_phi, rechit_layer, rechit_energy, rechit_time, seed_eta, seed_phi)

    # reduced4dim = np.sum(np.abs(pic3d), axis=3)
    #
    # xy_heatmap = np.sum(reduced4dim, axis=2)
    # xlayer_heatmap = np.sum(reduced4dim, axis=1)
    # ylayer_heatmap = np.sum(reduced4dim, axis=0)
    #
    # plt.imshow(xy_heatmap, cmap='hot', interpolation='nearest')
    # plt.show()
    #
    # plt.imshow(xlayer_heatmap, cmap='hot', interpolation='nearest')
    # plt.show()
    #
    # plt.imshow(ylayer_heatmap, cmap='hot', interpolation='nearest')
    # plt.show()

    print(np.shape(pic3d))


