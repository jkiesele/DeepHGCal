/*
 * ntuple_recHits.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_NTUPLE_RECHITS_H_
#define INCLUDE_NTUPLE_RECHITS_H_

#include "NTupleContent.h"

#define MAX_RECHITS 20000

class NTupleRecHits : public NTupleContent{
public:
	NTupleRecHits(const TString& id=""):NTupleContent(),id_(id),n_rechits_(0),nrechits_(0){}

	void initDNNBranches(TTree* t);

	void reset(){
		nrechits_=0;
		n_rechits_=0;
	}

    bool addRecHit(const float &eta,
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
                   const int &recHitParticle);


private:


	//DNN branches
	TString id_;
	int n_rechits_;
	float nrechits_;
	float rechit_energy_[MAX_RECHITS];
	float rechit_eta_[MAX_RECHITS];
	float rechit_phi_[MAX_RECHITS];
	float rechit_a_[MAX_RECHITS];
	float rechit_b_[MAX_RECHITS];
	float rechit_x_[MAX_RECHITS];
	float rechit_y_[MAX_RECHITS];
	float rechit_z_[MAX_RECHITS];
	float rechit_pt_[MAX_RECHITS];
	float rechit_time_[MAX_RECHITS];
	float rechit_layer_[MAX_RECHITS];
	float rechit_fraction_[MAX_RECHITS];
    int rechit_particle_[MAX_RECHITS];
    float rechit_seeddr_[MAX_RECHITS];
};


#endif /* INCLUDE_NTUPLE_RECHITS_H_ */
