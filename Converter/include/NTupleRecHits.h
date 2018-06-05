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

		rechit_energy_.clear();
		rechit_eta_.clear();
		rechit_phi_.clear();
		rechit_a_.clear();
		rechit_b_.clear();
		rechit_x_.clear();
		rechit_y_.clear();
		rechit_z_.clear();
		rechit_pt_.clear();
		rechit_time_.clear();
		rechit_layer_.clear();
		rechit_total_fraction_.clear();


	}

    bool addRecHit(const float &eta, const float &phi, const float &a, const float &b, const float &x,
                       const float &y, const float &z, const float &pt, const float &energy, const float &time,
                       const int &layer, const float &recHitTotalFraction);


private:


	//DNN branches
	TString id_;
	int n_rechits_;
	float nrechits_;
	std::vector<float> rechit_energy_;
	std::vector<float> rechit_eta_;
	std::vector<float> rechit_phi_;
	std::vector<float> rechit_a_;
	std::vector<float> rechit_b_;
	std::vector<float> rechit_x_;
	std::vector<float> rechit_y_;
	std::vector<float> rechit_z_;
	std::vector<float> rechit_pt_;
	std::vector<float> rechit_time_;
	std::vector<float> rechit_layer_;
	std::vector<float> rechit_total_fraction_;
};


#endif /* INCLUDE_NTUPLE_RECHITS_H_ */
