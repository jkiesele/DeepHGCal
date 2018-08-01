import numpy as np
import os


path = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/RelValTTbar_14TeV_CMSSW_10_0_0_pre1-PU25ns_94X_upgrade2023_realistic_v2_2023D17PU200-v1_GEN-SIM-RECO_NTUP/converted/bla_bla'

for dump_file in os.listdir(path):
    file_in = os.path.join(path, dump_file)
    with open(file_in, 'rb') as f:
        all_features = np.fromfile(f, dtype=np.float64)
        all_features = np.reshape(all_features, (-1, 6))

        print(all_features)