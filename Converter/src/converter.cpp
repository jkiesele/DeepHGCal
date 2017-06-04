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

#include "TFile.h"

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

    fChain->SetBranchStatus("multiclus_eta",1);
    fChain->SetBranchStatus("multiclus_phi",1);
    fChain->SetBranchStatus("multiclus_energy",1);


    fChain->SetBranchStatus("genpart_eta",1);
    fChain->SetBranchStatus("genpart_phi",1);
    fChain->SetBranchStatus("genpart_energy",1);
    fChain->SetBranchStatus("genpart_pid",1);

    fChain->SetBranchStatus("rechit_eta",1);
    fChain->SetBranchStatus("rechit_phi",1);
    fChain->SetBranchStatus("rechit_energy",1);
    fChain->SetBranchStatus("rechit_time",1);
    fChain->SetBranchStatus("rechit_layer",1);




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

    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        nb = fChain->GetEntry(jentry);   nbytes += nb;



        //create the seeds per event
        seedMaker seedmaker;
        seedmaker.createSeedsFromSimClusters(simcluster_eta,simcluster_phi);

        for(size_t i_seed=0;i_seed<seedmaker.seeds().size();i_seed++){

            globals.reset();
            recHits.reset();

            const seed& s=seedmaker.seeds().at(i_seed);
            globals.setSeedInfo(s.eta(),s.phi(),i_seed,jentry);

            //get truth
            bool hastruthmatch=false;
            int truthid=0;

            //change to look for the best truth match, not only any within the cone
            for(size_t i_t=0;i_t<genpart_eta->size();i_t++){

                //change to best match here
                if(s.matches(genpart_eta->at(i_t),genpart_phi->at(i_t), 0.02 )){

                    hastruthmatch=true;
                    globals.setTruthKinematics(genpart_eta->at(i_t),
                            genpart_phi->at(i_t),genpart_energy->at(i_t));

                    truthid=genpart_pid->at(i_t);
                    break;
                }
            }
            globals.setTruthID(truthid,hastruthmatch);

            for(size_t i_m=0;i_m<multiclus_eta->size();i_m++){
                if(s.matches(multiclus_eta->at(i_m), multiclus_phi->at(i_m),0.02)){
                    globals.setMultiCluster(multiclus_eta->at(i_m),
                            multiclus_phi->at(i_m),
                            multiclus_energy->at(i_m));
                    break;
                }
            }

            //match the recHits
            for(size_t i_r=0;i_r<rechit_eta->size();i_r++){
                if(s.matches(rechit_eta->at(i_r),rechit_phi->at(i_r), 0.2 )){
                    recHits.addRecHit(rechit_eta->at(i_r),rechit_phi->at(i_r),
                            rechit_energy->at(i_r),rechit_time->at(i_r),
                            rechit_layer->at(i_r));

                }
            }

            outtree->Fill();
        }


        /// do the selection and filling



    }

    outtree->Write();
    outfile->Close();
    delete outfile;
}
