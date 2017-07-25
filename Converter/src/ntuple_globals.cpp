/*
 * ntuple_globals.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */




#include "../include/ntuple_globals.h"

ntuple_globals::ntuple_globals():
ntuple_content()
{reset();}


void ntuple_globals::reset(){
	event_=0;

	seed_eta_=0;
	seed_phi_=0;
	seed_index_=0;

    totalrechit_energy_=0;
    true_energyfraction_=0;

    multicluster_eta_=0;
    multicluster_phi_=0;
    multicluster_energy_=0;

    simcluster_eta_=0;
    simcluster_phi_=0;
    simcluster_energy_=0;

    true_ncloseparticles_=0;
    true_drclosestparticle_=-1;
    true_averagedr_=0;
    true_closeparticledr_.clear();

	true_eta_=0;
	true_phi_=0;
	true_energy_=0;
	true_pid_=0;

    isGamma_=0;
    isElectron_=0;
    isMuon_=0;
    isTau_=0;
    isPionZero_=0;
    isPionCharged_=0;
    isEta_=0;
    isProton_=0;
    isKaonCharged_=0;
    isOther_=0;
    isFake_=0;

	true_vtx_x_=0;
	true_vtx_y_=0;
	true_vtx_z_=0;

	nmulticlusters_=0;
}

void ntuple_globals::setTruthID(const int& pdgid,const bool& matched){

    isGamma_=0;
    isElectron_=0;
    isMuon_=0;
    isTau_=0;
    isPionZero_=0;
    isPionCharged_=0;
    isEta_=0;
    isOther_=0;
    isFake_=0;
    true_pid_=0;
    isProton_=0;
    isKaonCharged_=0;

	if(!matched){
		isFake_=1;
		return;
	}
	true_pid_=pdgid;
	//some translation here - just example for now
	if(pdgid==22)
		isGamma_=1;
	else if(std::abs(pdgid)==11)
		isElectron_=1;
    else if(std::abs(pdgid)==13)
        isMuon_=1;
    else if(std::abs(pdgid)==15)
        isTau_=1;
    else if(std::abs(pdgid)==111)
        isPionZero_=1;
    else if(std::abs(pdgid)==211)
        isPionCharged_=1;
    else if(std::abs(pdgid)==221)
        isEta_=1;
    else if(pdgid==2212)
        isProton_=1;
    else if(std::abs(pdgid)==321)
        isKaonCharged_=1;
    else
        isOther_=1;

}



void ntuple_globals::initDNNBranches(TTree* t){


	addBranch(t,"event", &event_, "event_/i");

	addBranch(t,"seed_eta",   &seed_eta_,   "seed_eta_/f");
	addBranch(t,"seed_phi",   &seed_phi_,   "seed_phi_/f");
	addBranch(t,"seed_index", &seed_index_, "seed_index_/i");

	addBranch(t,"totalrechit_energy",   &totalrechit_energy_,   "totalrechit_energy_/f");
    addBranch(t,"true_energyfraction",   &true_energyfraction_,   "true_energyfraction_/f");

	addBranch(t,"multicluster_eta",   &multicluster_eta_,   "multicluster_eta_/f");
	addBranch(t,"multicluster_phi",   &multicluster_phi_,   "multicluster_phi_/f");
	addBranch(t,"multicluster_energy",   &multicluster_energy_,   "multicluster_energy_/f");

    addBranch(t,"simcluster_eta",      &simcluster_eta_,      "simcluster_eta_/f");
    addBranch(t,"simcluster_phi",      &simcluster_phi_,      "simcluster_phi_/f");
    addBranch(t,"simcluster_energy",   &simcluster_energy_,   "simcluster_energy_/f");


    addBranch(t,"true_ncloseparticles",   &true_ncloseparticles_,   "true_ncloseparticles_");
    addBranch(t,"true_drclosestparticle",   &true_drclosestparticle_,   "true_drclosestparticle_/f");
    addBranch(t,"true_averagedr",   &true_averagedr_,   "true_averagedr_/f");
    addBranch(t,"true_closeparticledr",   &true_closeparticledr_);


	addBranch(t,"true_eta",   &true_eta_,   "true_eta_/f");
	addBranch(t,"true_phi",   &true_phi_,   "true_phi_/f");
	addBranch(t,"true_energy",&true_energy_,"true_energy_/f");
	addBranch(t,"true_pid",&true_pid_,"true_pid_/i");


	addBranch(t,"isGamma",   &isGamma_,   "isGamma_/i");
    addBranch(t,"isElectron",   &isElectron_,   "isElectron_/i");
    addBranch(t,"isMuon",   &isMuon_,   "isMuon_/i");
    addBranch(t,"isTau",   &isTau_,   "isTau_/i");
    addBranch(t,"isPionZero",   &isPionZero_,   "isPionZero_/i");
    addBranch(t,"isPionCharged",   &isPionCharged_,   "isPionCharged_/i");
    addBranch(t,"isProton",   &isProton_,   "isProton_/i");
    addBranch(t,"isKaonCharged",   &isKaonCharged_,   "isKaonCharged_/i");
    addBranch(t,"isEta",   &isEta_,   "isEta_/i");
    addBranch(t,"isOther",   &isOther_,   "isOther_/i");
	addBranch(t,"isFake",    &isFake_,    "isFake_/i");

	addBranch(t,"true_vtx_x",   &true_vtx_x_,   "true_vtx_x_/f");
	addBranch(t,"true_vtx_y",   &true_vtx_y_,   "true_vtx_y_/f");
	addBranch(t,"true_vtx_z",   &true_vtx_z_,   "true_vtx_z_/f");


}

