/*
 * ntuple_recHits.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_NTUPLE_RECHITS_H_
#define INCLUDE_NTUPLE_RECHITS_H_

#include "ntuple_content.h"

#define MAX_RECHITS 2000

class ntuple_recHits : public ntuple_content{
public:
	ntuple_recHits(const TString& id=""):ntuple_content(),id_(id),nrechits_(0){}

	void initDNNBranches(TTree* t);

	void reset(){
		nrechits_=0;
	}

	void addRecHit(const float& eta,
			const float& phi,
			const float& energy,
			const float& time);


private:


	//DNN branches
	TString id_;
	int nrechits_;
	float rechit_energy_[MAX_RECHITS];
	float rechit_eta_[MAX_RECHITS];
	float rechit_phi_[MAX_RECHITS];
	float rechit_time_[MAX_RECHITS];

};


#endif /* INCLUDE_NTUPLE_RECHITS_H_ */
