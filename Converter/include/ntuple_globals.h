/*
 * ntuple_globals.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_NTUPLE_GLOBALS_H_
#define INCLUDE_NTUPLE_GLOBALS_H_

#include "ntuple_content.h"

class ntuple_globals : public ntuple_content{
public:
    ntuple_globals();

    void reset();

    void initDNNBranches(TTree* t);

    void setSeedInfo(const float& eta,
            const float& phi,
            const size_t& seedindex,const size_t& event){
        seed_eta_=eta;
        seed_phi_=phi;
        seed_index_=seedindex;
        event_=event;
    }

    void setCloseParticles(const float& eta,const float& phi,const float& maxdr){

        float dr2=(eta-seed_eta_)*(eta-seed_eta_)+(phi-seed_phi_)*(phi-seed_phi_);
        if(dr2>maxdr*maxdr || dr2==0) return;
        float dr=sqrt(dr2);
        float totav=true_averagedr_*true_ncloseparticles_;
        true_ncloseparticles_++;
        if((true_drclosestparticle_<0 || dr<true_drclosestparticle_) && dr)
            true_drclosestparticle_=dr;

        totav+=dr;
        true_averagedr_=totav/true_ncloseparticles_;
    }


    void setMultiCluster(const float& eta,
            const float& phi,
            const float& energy){

        float toteta=nmulticlusters_*multicluster_eta_;
        float totphi=nmulticlusters_*multicluster_phi_;
        toteta+=eta;
        totphi+=phi;
        nmulticlusters_++;
        multicluster_eta_=toteta/nmulticlusters_;
        multicluster_phi_=totphi/nmulticlusters_;
        multicluster_energy_+=energy;
    }

    void setTruthKinematics(const float& eta,
            const float& phi,
            const float& energy){
        true_eta_=eta;
        true_phi_=phi;
        true_energy_=energy;
    }

    void setSimCluster(const float& eta,
            const float& phi,
            const float& energy){
        simcluster_eta_=eta;
        simcluster_phi_=phi;
        simcluster_energy_=energy;
    }

    void setTotalRecHitEnergy(const float& e){
        totalrechit_energy_=e;
    }
    void computeTrueFraction(){
        true_energyfraction_=true_energy_/totalrechit_energy_;
    }

    void setTruthID(const int& pdgid,const bool& matched);


    //void setTruthVertex(); ///..later

private:


    //global info branches
    int event_;

    //DNN reco branches
    float seed_eta_;
    float seed_phi_;
    int   seed_index_;

    float totalrechit_energy_;
    float true_energyfraction_;

    //for comparison
    float multicluster_eta_;
    float multicluster_phi_;
    float multicluster_energy_;

    //for comparison
    float simcluster_eta_;
    float simcluster_phi_;
    float simcluster_energy_;

    //for performance evaluation
    float true_ncloseparticles_;
    float true_drclosestparticle_;
    float true_averagedr_;

    //DNN truth branches
    float true_eta_;
    float true_phi_;
    float true_energy_;

    int true_pid_;

    int isGamma_;
    int isElectron_;
    int isMuon_;
    int isTau_;
    int isPionZero_;
    int isPionCharged_;
    int isEta_;
    int isProton_;
    int isKaonCharged_;
    int isOther_;
    int isFake_;
    //to be extended

    //to be used only later
    float true_vtx_x_;
    float true_vtx_y_;
    float true_vtx_z_;


    // for temp use, only. Not to be stored
    float nmulticlusters_;


};


#endif /* INCLUDE_NTUPLE_GLOBALS_H_ */
