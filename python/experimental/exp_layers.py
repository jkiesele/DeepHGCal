import sparse_hgcal
import numpy as np

input_file = '/eos/cms/store/cmst3/group/dehep/miniCalo/prod2/2_15_out.root'
all_features, spatial, spatial_local, labels_one_hot, num_entries = sparse_hgcal.read_sparse_data(input_file, 3000)


