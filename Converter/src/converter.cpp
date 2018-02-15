/*
 * converter.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#include "../include/converter.h"
#include "../include/ntuple_globals.h"
#include "../include/ntuple_recHits.h"
#include "../include/seedMaker.h"
#include "../include/Transformer.h"
#include "../include/truthCreator.h"
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

#define LAYER_NUM 52
#define COORDINATE_A 0
#define COORDINATE_B 1


void converter::Loop(){


    //     This is the loop skeleton where:
    //    jentry is the global entry number in the chain
    //    ientry is the entry number in the current Tree
    //  Note that the argument to GetEntry must be:
    //    jentry for TChain::GetEntry
    //    ientry for TTree::GetEntry and TBranch::GetEntry
    //
    //       To read only selected branches, Insert statements like:
    // METHOD1:
    //    fChain->SetBranchStatus("*",0);  // disable all branches
    //    fChain->SetBranchStatus("branchname",1);  // activate branchname
    // METHOD2: replace line
    //    fChain->GetEntry(jentry);       //read all branches
    //by  b_branchname->GetEntry(ientry); //read only this branch
    if (fChain == 0) return;


    /////select branches
    fChain->SetBranchStatus("*",0);

    fChain->SetBranchStatus("event",1);
    fChain->SetBranchStatus("simcluster_eta",1);
    fChain->SetBranchStatus("simcluster_phi",1);
    fChain->SetBranchStatus("simcluster_energy",1);
    // Added by SRQ
    fChain->SetBranchStatus("simcluster_hits",1);
    fChain->SetBranchStatus("simcluster_fractions",1);

    fChain->SetBranchStatus("multiclus_eta",1);
    fChain->SetBranchStatus("multiclus_phi",1);
    fChain->SetBranchStatus("multiclus_energy",1);


    fChain->SetBranchStatus("genpart_eta",1);
    fChain->SetBranchStatus("genpart_phi",1);
    fChain->SetBranchStatus("genpart_ovz",1);
    fChain->SetBranchStatus("genpart_dvz",1);
    fChain->SetBranchStatus("genpart_energy",1);
    fChain->SetBranchStatus("genpart_pid",1);
    fChain->SetBranchStatus("genpart_reachedEE",1);
    fChain->SetBranchStatus("genpart_pt",1);


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



    //////create output file and tree
    if(outfilename_.Length() == 0) return;

    TFile * outfile=new TFile(outfilename_,"RECREATE");
    TDirectory * dir=outfile->mkdir("deepntuplizer","deepntuplizer");
    dir->cd();
    TTree* outtree=new TTree("tree","tree");

    //load DNN branches

    ntuple_globals globals;
    globals.initBranches(outtree);
    ntuple_recHits recHits;
    recHits.initBranches(outtree);

    Long64_t nentries = fChain->GetEntries();

   // Transformer transformer(rechit_x, rechit_y, rechit_z, this);

    int count = 0;
    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry < nentries;jentry++) {


        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        nb = fChain->GetEntry(jentry);   nbytes += nb;

        if(testmode_&&jentry>2) break;

        truthCreator truthcreator;
        std::vector<truthTarget> truth=truthcreator.createTruthTargets(genpart_eta,genpart_phi,genpart_energy,genpart_pt,
                genpart_ovz,genpart_dvz,
                genpart_pid);

        //create the seeds per event
        seedMaker seedmaker;
        // seedmaker.createSeedsFromCollection(genpart_eta,genpart_phi,genpart_reachedEE);
        seedmaker.createSeedsFromTruthTarget(truth);

        for(size_t i_seed=0;i_seed<seedmaker.seeds().size();i_seed++){
            bool write=true;

            globals.reset();
            recHits.reset();

            float DRaroundSeed=1;

            const seed& s=seedmaker.seeds().at(i_seed);
            globals.setSeedInfo(s.eta(),s.phi(),i_seed,jentry);

            //get truth
            bool hastruthmatch=false;
            int truthid=0;

            //change to look for the best truth match, not only any within the cone

            for(size_t i_t=0;i_t<truth.size();i_t++){

                //use best match here!
                float dr=s.matches(truth.at(i_t).eta(),truth.at(i_t).phi(), 10000 );
                if(dr && dr<DRaroundSeed){
                    if(s.truthIndex()==(int)i_t){
                        hastruthmatch=true;
                        globals.setTruthKinematics(truth.at(i_t).eta(),truth.at(i_t).phi(),
                                                         truth.at(i_t).energy(),truth.at(i_t).pt());
                        truthid=truth.at(i_t).pdgId();

                    }
                    else{
                        globals.setCloseParticles(truth.at(i_t).eta(),truth.at(i_t).phi(),DRaroundSeed);
                    }
                }
            }

            globals.setTruthID(truthid,hastruthmatch);
            if(globals.trueEnergy()<energylowercut_) continue;

            for(size_t i_m=0;i_m<multiclus_eta->size();i_m++){
                if(s.matches(multiclus_eta->at(i_m), multiclus_phi->at(i_m),0.05)){
                    globals.setMultiCluster(multiclus_eta->at(i_m),
                            multiclus_phi->at(i_m),
                            multiclus_energy->at(i_m));
                    //add all of them
                }
            }

            std::unordered_map<unsigned int, RecHitData> recHitsMap;

            // Iterate through rechits and put them in hashmap for faster search
            for(size_t i_r=0;i_r<rechit_eta->size();i_r++) {
                float eta = rechit_eta->at(i_r);
                float phi = rechit_phi->at(i_r);
                float energy = rechit_energy->at(i_r);
                unsigned int id = rechit_detid->at(i_r);
                RecHitData recHit = {id, eta, phi, energy};
                recHitsMap[id] = recHit;
            }
            // Iterate through all the sim clusters to recompute their eta and phi
            for(size_t i_m=0;i_m<simcluster_eta->size();i_m++) {
                std::vector<unsigned int> simClusterHitsIds = simcluster_hits->at(i_m);
                std::vector<float> simClusterHitsFractions = simcluster_fractions->at(i_m);

                float etaWeightedSum = 0;
                float phiWeightedSum = 0;
                float weightsSum = 0;
                for(size_t i = 0; i < simClusterHitsIds.size(); i++) {
                    unsigned int id = simClusterHitsIds[i];
                    float fraction = simClusterHitsFractions[i];

                    if (recHitsMap.find(id) == recHitsMap.end()) {
                        continue;
                    }

                    RecHitData recHit = recHitsMap[id]; // It should exist otherwise it will except
                    etaWeightedSum += recHit.eta * fraction * recHit.energy;
                    phiWeightedSum += helpers::deltaPhi(recHit.phi, 0) * fraction * recHit.energy;
                    weightsSum += fraction * recHit.energy;
                }


                simcluster_eta->at(i_m) = etaWeightedSum / weightsSum;
                simcluster_phi->at(i_m) = phiWeightedSum / weightsSum;
                if(etaWeightedSum==0) {
                    std::cerr << "Error in finding corresponding rechits for sim-clusters." << std::endl;
                    simcluster_eta->at(i_m) = simcluster_phi->at(i_m) = 10000000;
                }

            }


            if (simcluster_eta->size() == 0) {
                std::cerr<<"Error in simclusters"<<std::endl;
                exit(-1);
            }

            float* seedSimilarityVector = new float[simcluster_eta->size()];
            int simClusterIndex = 0;
            float minDistance = helpers::getSeedSimClusterDifference(s.eta(), s.phi(), simcluster_eta->at(0),
                                                                          simcluster_phi->at(0));
            seedSimilarityVector[0]=minDistance;
            for (size_t i_m = 1; i_m < simcluster_eta->size(); i_m++) {
                float newDistance = helpers::getSeedSimClusterDifference(s.eta(), s.phi(), simcluster_eta->at(i_m),
                                                                         simcluster_phi->at(i_m));
                if (newDistance < minDistance) {
                    minDistance = newDistance;
                    simClusterIndex = i_m;
                }
                seedSimilarityVector[i_m]=minDistance;
            }

            // ID to fraction
            float maxFraction = 0;
            std::unordered_map<unsigned int, float> fractions_map;
            {
                std::vector<unsigned int> simClusterHitsIds = simcluster_hits->at(simClusterIndex);
                std::vector<float> simClusterHitsFractions = simcluster_fractions->at(simClusterIndex);
                for(size_t i = 0; i <= simClusterHitsIds.size(); i++) {
                    fractions_map[simClusterHitsIds[i]] = simClusterHitsFractions[i];
                    maxFraction = max(maxFraction, simClusterHitsFractions[i]);
                }
            }

//            std::cout<<"Max fraction is "<<maxFraction<<std::endl;
            maxFraction = 0;
            // match the recHits
            float totalrechitenergy=0;
            for(size_t i_r=0;i_r<rechit_eta->size();i_r++){
                if(s.matches(rechit_eta->at(i_r),rechit_phi->at(i_r), DRaroundSeed )){

                	count ++;

                    unsigned int my_recthit_detid = rechit_detid->at(i_r);
                    auto fractions_map_iterator = fractions_map.find(my_recthit_detid);
                    float fraction;
                    float belongToASimCluster;
                    if (fractions_map_iterator != fractions_map.end()) {
                        fraction = fractions_map_iterator->second;
                        belongToASimCluster = 1;
                    }
                    else {
                        fraction = 0;
                        belongToASimCluster = 0;
                    }

                    if (fraction > 1)
                        fraction = 0;

                    maxFraction = max(maxFraction, fraction);

                    write &= recHits.addRecHit(
                	        rechit_eta->at(i_r),rechit_phi->at(i_r),
                	        0,0,
                	        rechit_x->at(i_r),rechit_y->at(i_r),
                	        rechit_z->at(i_r),
                	        rechit_pt->at(i_r), rechit_energy->at(i_r),
                	        rechit_time->at(i_r), rechit_layer->at(i_r),
                	        s.eta(),s.phi(), fraction, belongToASimCluster);

                	totalrechitenergy+=rechit_energy->at(i_r);

                }
            }

            globals.setTotalRecHitEnergy(totalrechitenergy);
            globals.computeTrueFraction();

            if(write)
                outtree->Fill();

        }
    }

    TCanvas canvas;

    outtree->SetMarkerColor(kGreen);
    outtree->Draw("rechit_eta:rechit_phi","event==0");
    outtree->SetMarkerColor(kYellow);
    outtree->Draw("rechit_eta:rechit_phi","event==0 && rechit_energy>0.1","same");
    outtree->SetMarkerColor(kOrange+8);
    outtree->Draw("rechit_eta:rechit_phi","event==0 && rechit_energy>0.3","same");
    outtree->SetMarkerColor(kRed);
    outtree->Draw("rechit_eta:rechit_phi","event==0 && rechit_energy>0.5","same");
    outtree->SetMarkerColor(kBlack);
    outtree->SetMarkerStyle(5);
    outtree->Draw("seed_eta:seed_phi","event==0","same");

    canvas.Write();

    outtree->Write();
    outfile->Close();
    delete outfile;
}


