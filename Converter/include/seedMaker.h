/*
 * seedMaker.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_SEEDMAKER_H_
#define INCLUDE_SEEDMAKER_H_


#include <cmath>
#include <vector>
#include <limits>
#include "helpers.h"
#include <string>
#include "truthCreator.h"
//helper

class seed{
public:
	seed(const float& eta, const float& phi):
	eta_(eta),phi_(phi),truthidx_(-1){}

	inline float matches(const float& eta, const float& phi, const float& dR)const{
		float deltaeta=eta_-eta;
		if(fabs(deltaeta)>dR)
			return 0;
		float deltaphi=helpers::deltaPhi(phi_,phi);
		if(fabs(deltaphi)>dR)
			return 0;

		float deltaR2=deltaeta*deltaeta + deltaphi*deltaphi;
		if(deltaR2<dR*dR){
			if(deltaR2)
				return sqrt(deltaR2);
			else
				return std::numeric_limits<float>::min();
		}
		return 0;

	}

	const float& eta()const{return eta_;}
	const float& phi()const{return phi_;}

	//only for seeds from truth. Gives index of truth particle in event
	const int truthIndex()const{return truthidx_;}
	void setTruthIndex(int idx){truthidx_=idx;}

private:
	float eta_,phi_;
	int truthidx_;
};

class seedMaker{
public:
	seedMaker():mineta(1.2),maxeta(3.){}

	void createSeedsFromCollection(const std::vector<float> *etas,
			const std::vector<float> *phis,
	        const std::vector<float> * select=0,
	        const float selectcut=0);

    void createMaxSeedsFromGenCollection(
            const std::vector<float> *etas,
            const std::vector<float> *phis,
            size_t max=1e6,
            const std::vector<int>*pidsel=0,
            int abspid=0);

    void createSeedsFromCollection(const std::vector<float> *etas,
            const std::vector<float> *phis,
            const std::vector<bool> * select);

    void createSeedsFromTruthTarget(const std::vector<truthTarget>&);

	const std::vector<seed>& seeds()const{return seeds_;}

	void clear(){
		seeds_.clear();
	}
private:
	std::vector<seed> seeds_;

	const float mineta,maxeta;

};



#endif /* INCLUDE_SEEDMAKER_H_ */
