//
// Created by Shah Rukh Qasim on 2/23/18.
//


#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

#ifndef DEEPHGCAL2_INDEXSIMCLUSTERS_H
#define DEEPHGCAL2_INDEXSIMCLUSTERS_H


class IndexSimClusters {
private:
    TTree* tree;

    std::vector<unsigned int> *rechit_detid;
    std::vector<std::vector<unsigned int>> *simcluster_hits;
    std::vector<std::vector<float>> *simcluster_fractions;

    TBranch *b_rechit_detid;
    TBranch *b_simcluster_hits;
    TBranch *b_simcluster_fractions;
public:
    IndexSimClusters(TTree *tree);
    void execute();
    void setBranches();

};


#endif //DEEPHGCAL2_INDEXSIMCLUSTERS_H
