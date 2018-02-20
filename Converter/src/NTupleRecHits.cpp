/*
 * ntuple_recHits.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#include "../include/NTupleRecHits.h"


#include <iostream>
#include <cmath>
#include "../include/helpers.h"


void NTupleRecHits::initDNNBranches(TTree* t){

	TString add="";
	if(id_.Length() && !id_.EndsWith("_"))
		add=id_+"_";

	addBranch(t,add+"n_rechits",&n_rechits_,"n_rechits_/i");

	addBranch(t,add+"nrechits",     &nrechits_,"nrechits_/f");
	addBranch(t,add+"rechit_eta",   &rechit_eta_,   "rechit_eta_[n_rechits_]/f");
	addBranch(t,add+"rechit_phi",   &rechit_phi_,   "rechit_phi_[n_rechits_]/f");
	addBranch(t,add+"rechit_a",   &rechit_a_,   "rechit_a_[n_rechits_]/f");
	addBranch(t,add+"rechit_b",   &rechit_b_,   "rechit_b_[n_rechits_]/f");
	addBranch(t,add+"rechit_x",   &rechit_x_,   "rechit_x_[n_rechits_]/f");
	addBranch(t,add+"rechit_y",   &rechit_y_,   "rechit_y_[n_rechits_]/f");
	addBranch(t,add+"rechit_z",   &rechit_z_,   "rechit_z_[n_rechits_]/f");
	addBranch(t,add+"rechit_energy",&rechit_energy_,"rechit_energy_[n_rechits_]/f");
	addBranch(t,add+"rechit_time"  ,&rechit_time_,  "rechit_time_[n_rechits_]/f");
	addBranch(t,add+"rechit_layer"  ,&rechit_layer_,  "rechit_layer_[n_rechits_]/f");
    addBranch(t,add+"rechit_seeddr"  ,&rechit_seeddr_,  "rechit_seeddr_[n_rechits_]/f");
	addBranch(t,add+"rechit_fraction"  ,&rechit_fraction_,  "rechit_fraction_[n_rechits_]/f");
	addBranch(t,add+"rechit_particle"  ,&rechit_particle_,  "rechit_particle[n_rechits_]/i");


}


bool NTupleRecHits::addRecHit(const float &eta,
							   const float &phi,
							   const float &a,
							   const float &b,
							   const float &x,
							   const float &y,
							   const float &z,
							   const float &pt,
							   const float &energy,
							   const float &time,
							   const int &layer,
							   const float &seedeta,
							   const float &seedphi,
							   const float &recHitFraction,
							   const int &recHitParticle) {

	if (n_rechits_ >= MAX_RECHITS) {
		std::cout << "WARNING: MAX NUMBER OF REC HITS REACHED" << std::endl;
		return false;
	}
	rechit_eta_[n_rechits_] = eta;
	rechit_phi_[n_rechits_] = phi;
	rechit_a_[n_rechits_] = a;
	rechit_b_[n_rechits_] = b;
	rechit_x_[n_rechits_] = x;
	rechit_y_[n_rechits_] = y;
	rechit_z_[n_rechits_] = z;
	rechit_pt_[n_rechits_] = pt;
	rechit_energy_[n_rechits_] = energy;
	rechit_time_[n_rechits_] = time;
	rechit_layer_[n_rechits_] = layer;
	float deltaphi = helpers::deltaPhi(phi, seedphi);
	rechit_seeddr_[n_rechits_] = std::sqrt((eta - seedeta) * (eta - seedeta) + deltaphi * deltaphi);
	rechit_fraction_[n_rechits_] = recHitFraction;
	rechit_particle_[n_rechits_] = recHitParticle;

	nrechits_++;
	n_rechits_++;

	return true;
}
