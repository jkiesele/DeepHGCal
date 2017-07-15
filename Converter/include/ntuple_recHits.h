/*
 * ntuple_recHits.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_NTUPLE_RECHITS_H_
#define INCLUDE_NTUPLE_RECHITS_H_

#include "ntuple_content.h"

#define MAX_RECHITS 20000

class ntuple_recHits : public ntuple_content{
public:
	ntuple_recHits(const TString& id=""):ntuple_content(),id_(id),n_rechits_(0),nrechits_(0){}

	void initDNNBranches(TTree* t);

	void reset(){
		nrechits_=0;
		n_rechits_=0;
	}

	void addRecHit(const float& eta,
			const float& phi,
			const float& energy,
			const float& time,
			const int& layer,
            const float& seedeta,
            const float& seedphi);


private:


	//DNN branches
	TString id_;
	int n_rechits_;
	float nrechits_;
	float rechit_energy_[MAX_RECHITS];
	float rechit_eta_[MAX_RECHITS];
	float rechit_phi_[MAX_RECHITS];
	float rechit_time_[MAX_RECHITS];
	float rechit_layer_[MAX_RECHITS];

    float rechit_seeddr_[MAX_RECHITS];

};


#endif /* INCLUDE_NTUPLE_RECHITS_H_ */
