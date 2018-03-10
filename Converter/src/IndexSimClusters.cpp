//
// Created by Shah Rukh Qasim on 2/23/18.
//

#include "../include/IndexSimClusters.h"
#include <iostream>

using namespace std;

IndexSimClusters::IndexSimClusters(TTree *tree) : tree(tree) {

}

void IndexSimClusters::setBranches() {
    tree->SetBranchAddress("rechit_detid", &rechit_detid, &b_rechit_detid);
    tree->SetBranchAddress("simcluster_hits", &simcluster_hits, &b_simcluster_hits);
    tree->SetBranchAddress("simcluster_fractions", &simcluster_fractions, &b_simcluster_fractions);

    tree->SetBranchStatus("*",0);
    tree->SetBranchStatus("rechit_detid",1);
    tree->SetBranchStatus("simcluster_hits",1);
    tree->SetBranchStatus("simcluster_fractions",1);
}

void IndexSimClusters::execute() {
    setBranches();


    Long64_t numEntries = tree->GetEntries();

    for (Long64_t i_entry = 0; i_entry < numEntries; i_entry++) {
        if(!tree->LoadTree(i_entry))
            break;
        cout<<"Hello, world!";
    }
}
