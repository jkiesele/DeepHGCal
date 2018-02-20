//
// Created by srq2 on 2/20/18.
//

#include "../include/NTupleSimClusters.h"


void NTupleSimClusters::initDNNBranches(TTree *t) {
    TString add="";

    addBranch(t,add+"n_rechits",&nrechits_,"n_rechits_/i");
    addBranch(t,add+"rechit_eta",   &rechit_eta_,   "rechit_eta_[n_rechits_]/f");
    addBranch(t,add+"rechit_phi",   &rechit_phi_,   "rechit_phi_[n_rechits_]/f");
    addBranch(t,add+"rechit_x",   &rechit_x_,   "rechit_x_[n_rechits_]/f");
    addBranch(t,add+"rechit_y",   &rechit_y_,   "rechit_y_[n_rechits_]/f");
    addBranch(t,add+"rechit_z",   &rechit_z_,   "rechit_z_[n_rechits_]/f");
    addBranch(t,add+"rechit_energy",&rechit_energy_,"rechit_energy_[n_rechits_]/f");
    addBranch(t,add+"rechit_time"  ,&rechit_time_,  "rechit_time_[n_rechits_]/f");

    addBranch(t,add+"nsimclsuters",&nsimclusters_,"n_rechits_/i");
    addBranch(t,add+"simcluster_id",   &simcluster_id_,   "simcluster_id_[n_rechits_]/f");
    addBranch(t,add+"simcluster_eta",   &simcluster_eta_,   "simcluster_eta_[n_rechits_]/f");
    addBranch(t,add+"simcluster_phi",   &simcluster_phi_,   "simcluster_phi_[n_rechits_]/f");

}