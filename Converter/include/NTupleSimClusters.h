//
// Created by srq2 on 2/20/18.
//

#ifndef DEEPHGCAL2_NTUPLESIMCLUSTERS_H
#define DEEPHGCAL2_NTUPLESIMCLUSTERS_H


#include "NTupleContent.h"

#define MAX_RECHITS 40000
#define MAX_SIMCLUSTERS 500

class NTupleSimClusters : public NTupleContent {
public:
    void initDNNBranches(TTree* t);
    void addRecHit(float rechit_eta, float rechit_phi, float rechit_energy, float rechit_time, float rechit_x,
                    float rechit_y, float rechit_z);
private:
    float nsimclusters_;
    float simcluster_id_[MAX_SIMCLUSTERS];
    float simcluster_eta_[MAX_SIMCLUSTERS];
    float simcluster_phi_[MAX_SIMCLUSTERS];

    float nrechits_;
    float rechit_energy_[MAX_RECHITS];
    float rechit_eta_[MAX_RECHITS];
    float rechit_phi_[MAX_RECHITS];
    float rechit_x_[MAX_RECHITS];
    float rechit_y_[MAX_RECHITS];
    float rechit_z_[MAX_RECHITS];
    float rechit_time_[MAX_RECHITS];


};


#endif //DEEPHGCAL2_NTUPLESIMCLUSTERS_H
