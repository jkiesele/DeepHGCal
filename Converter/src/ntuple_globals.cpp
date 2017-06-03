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

	true_eta_=0;
	true_phi_=0;
	true_energy_=0;

	isGamma_=0;
	isHadron_=0;
	isFake_=0;

	true_vtx_x_=0;
	true_vtx_y_=0;
	true_vtx_z_=0;
}

void ntuple_globals::setTruthID(const int& pdgid,const bool& matched){
	isGamma_=0;
	isHadron_=0;
	isFake_=0;
	if(!matched){
		isFake_=1;
		return;
	}
	//some translation here - just example for now
	if(pdgid==22)
		isGamma_=1;
	else
		isHadron_=1;


}



void ntuple_globals::initDNNBranches(TTree* t){


	addBranch(t,"event", &event_, "event_/i");

	addBranch(t,"seed_eta",   &seed_eta_,   "seed_eta_/f");
	addBranch(t,"seed_phi",   &seed_phi_,   "seed_phi_/f");
	addBranch(t,"seed_index", &seed_index_, "seed_index_/i");

	addBranch(t,"true_eta",   &true_eta_,   "true_eta_/f");
	addBranch(t,"true_phi",   &true_phi_,   "true_phi_/f");
	addBranch(t,"true_energy",&true_energy_,"true_energy_/f");

	addBranch(t,"isGamma",   &isGamma_,   "isGamma_/i");
	addBranch(t,"isHadron",  &isHadron_,  "isHadron_/i");
	addBranch(t,"isFake",    &isFake_,    "isFake_/i");

	addBranch(t,"true_vtx_x",   &true_vtx_x_,   "true_vtx_x_/f");
	addBranch(t,"true_vtx_y",   &true_vtx_y_,   "true_vtx_y_/f");
	addBranch(t,"true_vtx_z",   &true_vtx_z_,   "true_vtx_z_/f");


}

