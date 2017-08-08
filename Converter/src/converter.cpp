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

#include "TCanvas.h"
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
    fChain->SetBranchStatus("simcluster_energy",1);

    fChain->SetBranchStatus("multiclus_eta",1);
    fChain->SetBranchStatus("multiclus_phi",1);
    fChain->SetBranchStatus("multiclus_energy",1);


    fChain->SetBranchStatus("genpart_eta",1);
    fChain->SetBranchStatus("genpart_phi",1);
    fChain->SetBranchStatus("genpart_energy",1);
    fChain->SetBranchStatus("genpart_pid",1);
    fChain->SetBranchStatus("genpart_reachedEE",1);

    fChain->SetBranchStatus("rechit_eta",1);
    fChain->SetBranchStatus("rechit_phi",1);
    fChain->SetBranchStatus("rechit_x",1);
    fChain->SetBranchStatus("rechit_y",1);
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

        if(testmode_&&jentry>50) break;

        //create the seeds per event
        seedMaker seedmaker;
        seedmaker.createSeedsFromCollection(genpart_eta,genpart_phi,genpart_reachedEE);

        for(size_t i_seed=0;i_seed<seedmaker.seeds().size();i_seed++){

            globals.reset();
            recHits.reset();

            float DRaroundSeed=0.35;

            const seed& s=seedmaker.seeds().at(i_seed);
            globals.setSeedInfo(s.eta(),s.phi(),i_seed,jentry);

            //get truth
            bool hastruthmatch=false;
            int truthid=0;

            //change to look for the best truth match, not only any within the cone
            float lasttruedr=10;

            for(size_t i_t=0;i_t<genpart_eta->size();i_t++){

                //use best match here!
                float dr=s.matches(genpart_eta->at(i_t),genpart_phi->at(i_t), 10000 );
                if(dr && dr<DRaroundSeed){
                    if(lasttruedr>dr){
                        hastruthmatch=true;
                        globals.setTruthKinematics(genpart_eta->at(i_t),
                                genpart_phi->at(i_t),genpart_energy->at(i_t));

                        truthid=genpart_pid->at(i_t);
                        lasttruedr=dr;
                    }
                    //break;
                }
            }
            for(size_t i_t=0;i_t<genpart_eta->size();i_t++){
                globals.setCloseParticles(genpart_eta->at(i_t),genpart_phi->at(i_t),DRaroundSeed);
            }

            globals.setTruthID(truthid,hastruthmatch);

            for(size_t i_m=0;i_m<multiclus_eta->size();i_m++){
                if(s.matches(multiclus_eta->at(i_m), multiclus_phi->at(i_m),0.05)){
                    globals.setMultiCluster(multiclus_eta->at(i_m),
                            multiclus_phi->at(i_m),
                            multiclus_energy->at(i_m));
                    //add all of them
                }
            }

            for(size_t i_m=0;i_m<simcluster_eta->size();i_m++){
                if(s.matches(simcluster_eta->at(i_m), simcluster_phi->at(i_m),0.003)){
                    globals.setSimCluster(simcluster_eta->at(i_m),
                            simcluster_phi->at(i_m),
                            simcluster_energy->at(i_m));
                    break;
                }
            }

            //match the recHits
            Transformer transformer(rechit_x, rechit_y);
            float totalrechitenergy=0;
            for(size_t i_r=0;i_r<rechit_eta->size();i_r++){
                if(s.matches(rechit_eta->at(i_r),rechit_phi->at(i_r), DRaroundSeed )){
                	vector<float> trans = transformer.transform(rechit_x->at(i_r), rechit_y->at(i_r));
                    recHits.addRecHit(rechit_eta->at(i_r),rechit_phi->at(i_r),
                    		trans[0], trans[1],
                            rechit_energy->at(i_r),rechit_time->at(i_r),
                            rechit_layer->at(i_r),
                            s.eta(),s.phi());
                    totalrechitenergy+=rechit_energy->at(i_r);

                }
            }
            globals.setTotalRecHitEnergy(totalrechitenergy);
            globals.computeTrueFraction();

            outtree->Fill();
        }


        /// do the selection and filling



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
