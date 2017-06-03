/*
 * ntuple_recHits.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#include "../include/ntuple_recHits.h"


#include <iostream>


void ntuple_recHits::initDNNBranches(TTree* t){

	TString add="";
	if(id_.Length() && !id_.EndsWith("_"))
		add=id_+"_";

	addBranch(t,add+"n_rechits",&n_rechits_,"n_rechits_/i");

	addBranch(t,add+"nrechits",     &nrechits_,"nrechits_/f");
	addBranch(t,add+"rechit_eta",   &rechit_eta_,   "rechit_eta_[n_rechits_]/f");
	addBranch(t,add+"rechit_phi",   &rechit_phi_,   "rechit_phi_[n_rechits_]/f");
	addBranch(t,add+"rechit_energy",&rechit_energy_,"rechit_energy_[n_rechits_]/f");
	addBranch(t,add+"rechit_time"  ,&rechit_time_,  "rechit_time_[n_rechits_]/f");
	addBranch(t,add+"rechit_layer"  ,&rechit_layer_,  "rechit_layer_[n_rechits_]/i");


}


void ntuple_recHits::addRecHit(const float& eta,
			const float& phi,
			const float& energy,
			const float& time,
			const int& layer){

		if(n_rechits_>=MAX_RECHITS){
			std::cout << "WARNING: MAX NUMBER OF REC HITS REACHED" << std::endl;
			return;
		}
		rechit_eta_[n_rechits_]=eta;
		rechit_phi_[n_rechits_]=phi;
		rechit_energy_[n_rechits_]=energy;
		rechit_time_[n_rechits_]=time;
		rechit_layer_[n_rechits_]=layer;
		nrechits_++;
		n_rechits_++;

	}
