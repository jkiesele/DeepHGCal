/*
 * converter.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#include "../include/Converter.h"
#include "../include/NTupleGlobals.h"
#include "../include/NTupleRecHits.h"
#include <stdlib.h>

#include "TCanvas.h"
#include "TFile.h"

#include <iostream>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <math.h>
#include <TH1F.h>
#include <TGraph.h>
#include "../include/helpers.h"


Converter::Converter(TTree *t, bool noIndices) : HGCalSel(t),testmode_(false),energylowercut_(0), noIndices(noIndices){

}

void Converter::traceDecayTree(unordered_set<int> &decayParticlesCluster, unordered_set<int> &allParticles) const {
    if (decayParticlesCluster.size() == 0)
        return;

    while(true) {
        bool doBreak = true;
        unordered_set<int> toBeRemoved;
        for(auto i : allParticles) {
            if (decayParticlesCluster.find(i) != decayParticlesCluster.end())
                continue;

            int mother = genpart_mother->at(i);

            if(mother < 0 ) {
                continue;
            }

            float mDecayX = genpart_dvx->at(mother);
            float mDecayY = genpart_dvy->at(mother);
            float mDecayZ = genpart_dvz->at(mother);

            float cOriginX = genpart_ovx->at(i);
            float cOriginY = genpart_ovy->at(i);
            float cOriginZ = genpart_ovz->at(i);

            float diff = sqrt(pow(mDecayX - cOriginX,2) + pow(mDecayY - cOriginY,2) + pow(mDecayZ - cOriginZ,2));

            if(decayParticlesCluster.find(mother) != decayParticlesCluster.end()) {
                decayParticlesCluster.insert(i);
                doBreak = false;
                toBeRemoved.insert(i);
            }
        }

        for(auto i : toBeRemoved) {
            allParticles.erase(i);
        }

        if (doBreak)
            break;
    }

}


unordered_map<int, pair<vector<int>, vector<int>>> Converter::findParticlesFromCollision() const {

    /*
     * Collision occurs at z = 0. Boundary A is very close to it to see each particle originating from collision.
     * Boundary B is close to close to the calorimeter. Between A and B, particle breaks into many sub-particles.
     * And each of these particles will likely generate a sim-cluster.
     *
     */

    // Index to daughters
    unordered_map<int, pair<vector<int>, vector<int>>> interestingParticles;
    unordered_set<int> allParticles;

    // Find all truth particles
    for(size_t i=0;i<genpart_eta->size();i++) {
        allParticles.insert(i);
        if (fabs(genpart_ovz->at(i)) >= ZPLANE_CUT)
            continue;
        if (fabs(genpart_dvz->at(i)) < ZPLANE_CUT)
            continue;

        interestingParticles[i] = pair<vector<int>, vector<int>>();
    }

    for(auto &i : interestingParticles) {
        unordered_set<int> decayParticles;
        decayParticles.insert(i.first);
        traceDecayTree(decayParticles, allParticles);
        vector<int> particles;
        vector<int> calorimeterParticles;
        for(auto j : decayParticles) {
            particles.push_back(j);

            const float direction = genpart_eta->at(j) >= 0 ? 1. : -1.;

            if (fabs(genpart_ovz->at(j)) >= ZPLANE_CUT_CALORIMETER)
                continue;
            if (fabs(genpart_dvz->at(j)) < ZPLANE_CUT_CALORIMETER)
                continue;

            calorimeterParticles.push_back(j);
        }

        i.second = pair<vector<int>, vector<int>>(particles, calorimeterParticles);
    }


    return interestingParticles;

}


void Converter::initializeBranches() {
    if (fChain == 0) return;

    fChain->SetBranchStatus("*",0);

    fChain->SetBranchStatus("event",1);
    fChain->SetBranchStatus("simcluster_eta",1);
    fChain->SetBranchStatus("simcluster_phi",1);
    fChain->SetBranchStatus("simcluster_energy",1);
    fChain->SetBranchStatus("simcluster_hits",1);
    fChain->SetBranchStatus("simcluster_fractions",1);

    if (not noIndices)
        fChain->SetBranchStatus("simcluster_hits_indices",1);

    fChain->SetBranchStatus("multiclus_eta",1);
    fChain->SetBranchStatus("multiclus_phi",1);
    fChain->SetBranchStatus("multiclus_energy",1);


    fChain->SetBranchStatus("genpart_eta",1);
    fChain->SetBranchStatus("genpart_phi",1);
    fChain->SetBranchStatus("genpart_exeta",1);
    fChain->SetBranchStatus("genpart_exphi",1);
    fChain->SetBranchStatus("genpart_exx",1);
    fChain->SetBranchStatus("genpart_exy",1);
    fChain->SetBranchStatus("genpart_ovx",1);
    fChain->SetBranchStatus("genpart_ovy",1);
    fChain->SetBranchStatus("genpart_ovz",1);
    fChain->SetBranchStatus("genpart_dvx",1);
    fChain->SetBranchStatus("genpart_dvy",1);
    fChain->SetBranchStatus("genpart_dvz",1);
    fChain->SetBranchStatus("genpart_energy",1);
    fChain->SetBranchStatus("genpart_pid",1);
    fChain->SetBranchStatus("genpart_reachedEE",1);
    fChain->SetBranchStatus("genpart_pt",1);
    fChain->SetBranchStatus("genpart_mother",1);


    fChain->SetBranchStatus("rechit_eta",1);
    fChain->SetBranchStatus("rechit_phi",1);
    fChain->SetBranchStatus("rechit_x",1);
    fChain->SetBranchStatus("rechit_y",1);
    fChain->SetBranchStatus("rechit_z",1);
    fChain->SetBranchStatus("rechit_pt",1);
    fChain->SetBranchStatus("rechit_energy",1);
    fChain->SetBranchStatus("rechit_time",1);
    fChain->SetBranchStatus("rechit_layer",1);
    fChain->SetBranchStatus("rechit_detid",1);
}


void Converter::addParticleDataToGlobals(NTupleGlobals &globals, size_t index) {

    float r_origin = helpers::cartesianToSphericalR(genpart_ovx->at(index), genpart_ovy->at(index),
                                                    genpart_ovz->at(index));
    float r_decay = helpers::cartesianToSphericalR(genpart_dvx->at(index), genpart_dvy->at(index),
                                                    genpart_dvz->at(index));


    const float direction = genpart_eta->at(index) >= 0 ? 1. : -1.;

    float eta;
    float phi;
    float x;
    float y;
    float z;

    if (direction * genpart_dvz->at(index) < ZPLANE_CUT_CALORIMETER) {
        eta = genpart_eta->at(index);
        phi = genpart_phi->at(index);
        x = genpart_dvx->at(index);
        y = genpart_dvy->at(index);
        z = genpart_dvz->at(index);
    }
    else {
        eta = genpart_exeta->at(index);
        phi = genpart_exphi->at(index);
        x = genpart_exx->at(index);
        y = genpart_exy->at(index);
        z = direction * 320;
    }

    globals.addParticleData(eta, phi, r_origin, r_decay,
                            genpart_ovx->at(index), genpart_ovy->at(index), genpart_ovz->at(index),
                            x, y, z);

}

void Converter::recomputeSimClusterEtaPhi() {
    size_t numSimClusters = simcluster_hits->size();
    // Iterate through all the sim clusters to recompute their eta and phi
    for (size_t i_m = 0; i_m < numSimClusters; i_m++) {
        std::vector<unsigned int> simClusterHitsIds = simcluster_hits->at(i_m);
        std::vector<int> simClusterHitsIndices = simcluster_hits_indices->at(i_m);
        std::vector<float> simClusterHitsFractions = simcluster_fractions->at(i_m);

        float etaWeightedSum = 0;
        float phiWeightedSum = 0;
        float weightsSum = 0;
        for (size_t i = 0; i < simClusterHitsIds.size(); i++) {
            float fraction = simClusterHitsFractions[i];
            int indexRechit = simClusterHitsIndices[i];

            if (indexRechit < 0 or indexRechit > 7000000) {
                continue;
            }

            etaWeightedSum += rechit_eta->at(indexRechit) * fraction * rechit_energy->at(indexRechit);
            phiWeightedSum += helpers::deltaPhi(rechit_phi->at(indexRechit), 0) * fraction * rechit_energy->at(indexRechit);
            weightsSum += fraction * rechit_energy->at(indexRechit);
        }


        simcluster_eta->at(i_m) = etaWeightedSum / weightsSum;
        simcluster_phi->at(i_m) = phiWeightedSum / weightsSum;
        if (weightsSum == 0) {
            simcluster_eta->at(i_m) = simcluster_phi->at(i_m) = std::numeric_limits<float>::min();
        }
    }
}



unordered_map<int, int> Converter::findSimClusterForSeeds(vector<int>& seeds) {
    size_t numSimClusters = simcluster_hits->size();
    unordered_set<int> taken;
    unordered_map<int, int> simClustersForSeeds;

    for (auto i : seeds) {
        float seedEta = genpart_exeta->at(i);
        float seedPhi = genpart_exphi->at(i);

        float minDistance;
        int simClusterIndex = -1;


        for (size_t i_m = 0; i_m < numSimClusters; i_m++) {
//            if (taken.find(i_m) != taken.end())
//                continue;
            float newDistance = helpers::getSeedSimClusterDifference(seedEta, seedPhi,
                                                                     simcluster_eta->at(i_m),
                                                                     simcluster_phi->at(i_m))
                                + fabs(log(simcluster_energy->at(i_m) / genpart_energy->at(i))*0.02);
            if ((newDistance < minDistance && simClusterIndex != -1) || simClusterIndex == -1) {
                minDistance = newDistance;
                simClusterIndex = i_m;
            }
        }


        if (simClusterIndex != -1 && minDistance <= 0.002) {
            simClustersForSeeds[i] = simClusterIndex;
            taken.insert(simClusterIndex);
        }
    }
    return simClustersForSeeds;
}

Long64_t Converter::loadEvent(const Long64_t &eventNo) {
    Long64_t ientry = LoadTree(eventNo);
    if (ientry < 0) return ientry;
    fChain->GetEntry(eventNo);

    // Fill the hash map
    unordered_map<long, long> rechitIdsToIndices;
    for(size_t iRecHit = 0; iRecHit < rechit_eta->size(); iRecHit++) {
        rechitIdsToIndices[rechit_detid->at(iRecHit)] = iRecHit;
    }

    if (noIndices) {
        simcluster_hits_indices_computed.clear();
        size_t numSimClusters = simcluster_hits->size();
        // Iterate through all the sim clusters to recompute their eta and phi
        for (size_t iSimCluster = 0; iSimCluster < numSimClusters; iSimCluster++) {
            std::vector<unsigned int> simClusterHitsIds = simcluster_hits->at(iSimCluster);
            std::vector<int> simClusterHitsIndices(simClusterHitsIds.size());
            for (size_t j = 0; j < simClusterHitsIds.size(); j++) {
                auto hitIndex = rechitIdsToIndices.find(simClusterHitsIds[j]);
                if (hitIndex == rechitIdsToIndices.end())
                    simClusterHitsIndices[j] = -1;
                else
                    simClusterHitsIndices[j] = hitIndex->second;
            }

            simcluster_hits_indices_computed.push_back(simClusterHitsIndices);
        }

        simcluster_hits_indices = &simcluster_hits_indices_computed;
    }
}


void Converter::Loop(){
    initializeBranches();
    const float DR_AROUND_SEED = 0.15;

    // Create output file and tree
    if (outfilename_.Length() == 0)
        return;
    TFile *outfile = new TFile(outfilename_, "RECREATE");
    TDirectory *dir = outfile->mkdir("deepntuplizer", "deepntuplizer");
    dir->cd();
    TTree *outtree = new TTree("tree", "tree");

    //load DNN branches

    NTupleGlobals globals;
    globals.initBranches(outtree);
    NTupleRecHits recHits;
    recHits.initBranches(outtree);

    Long64_t nentries = fChain->GetEntries();

    for (Long64_t jentry = 0; jentry < nentries; jentry++) {
        if (loadEvent(jentry) < 0)
            break;

        if (testmode_ && jentry > 2) break;

        unordered_map<int, pair<vector<int>, vector<int>>> particlesFromCollision = findParticlesFromCollision();
        recomputeSimClusterEtaPhi();

        for (auto particleFromCollisionIterator : particlesFromCollision) {
            globals.reset();
            recHits.reset();

            globals.setTruthKinematics(genpart_eta->at(particleFromCollisionIterator.first),
                                       genpart_phi->at(particleFromCollisionIterator.first),
                                       genpart_energy->at(particleFromCollisionIterator.first),
                                       genpart_pt->at(particleFromCollisionIterator.first));

            globals.setTruthID(genpart_pid->at(particleFromCollisionIterator.first), true);
            globals.setSeedInfo(genpart_eta->at(particleFromCollisionIterator.first),
                                genpart_phi->at(particleFromCollisionIterator.first),
                                particleFromCollisionIterator.first,
                                jentry);

            // Add all decay tree
            for(auto daughterIndexIterator : particleFromCollisionIterator.second.first) {
                addParticleDataToGlobals(globals, daughterIndexIterator);
            }

            unordered_map<int, int> simClustersForSeeds = findSimClusterForSeeds(
                    particleFromCollisionIterator.second.second);

            // Filter if we couldn't match any simclusters
            bool matched = false;
            for (auto simClusterForSeed : simClustersForSeeds)
                matched = matched or (simClusterForSeed.second!=-1);
            if (not matched) {
                cout<<"Skipping a sample"<<endl;
                continue;
            }


            size_t numRecHits = rechit_eta->size();
            vector<float> totalFraction(numRecHits, 0);

            float totalRecHitsEnergy = 0;

            for(auto j : simClustersForSeeds) {
                std::vector<unsigned int> simClusterHitsIds = simcluster_hits->at(j.second);
                std::vector<int> recHitIndexes= simcluster_hits_indices->at(j.second);
                std::vector<float> simClusterHitsFractions = simcluster_fractions->at(j.second);

                for(size_t k = 0; k < recHitIndexes.size(); k++) {
                    if(recHitIndexes[k] < 0 or recHitIndexes[k] > 7000000)
                        continue;
                    totalFraction[recHitIndexes[k]] += simClusterHitsFractions[k];
                }
            }


            int count = 0, count2 = 0;
            for(size_t iRecHit = 0; iRecHit < numRecHits; iRecHit++) {
                if (!helpers::recHitMatchesParticle(genpart_eta->at(particleFromCollisionIterator.first),
                                                    genpart_phi->at(particleFromCollisionIterator.first),
                                                    rechit_eta->at(iRecHit), rechit_phi->at(iRecHit), DR_AROUND_SEED))
                    continue;

                recHits.addRecHit(rechit_eta->at(iRecHit), rechit_phi->at(iRecHit), 0, 0, rechit_x->at(iRecHit),
                                  rechit_y->at(iRecHit),
                                  rechit_z->at(iRecHit), rechit_pt->at(iRecHit), rechit_energy->at(iRecHit),
                                  rechit_time->at(iRecHit), rechit_layer->at(iRecHit), fmax(0,fmin(1,totalFraction.at(iRecHit))));

                totalRecHitsEnergy += rechit_energy->at(iRecHit);
                count += 1;
                if (fmax(0,fmin(1,totalFraction.at(iRecHit))) > 0.3)
                    count2 ++;

            }

            globals.setTotalRecHitEnergy(totalRecHitsEnergy);
            globals.computeTrueFraction();

            outtree->Fill();
        }

        cout<<"Done "<<jentry<<endl;
    }


    outtree->Write();
    outfile->Close();
    delete outfile;
}


