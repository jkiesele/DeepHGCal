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
//helper

class seed{
public:
	seed(const float& eta, const float& phi):
	eta_(eta),phi_(phi){}

	inline float matches(const float& eta, const float& phi, const float& dR)const{
		float deltaeta=eta_-eta;
		if(fabs(deltaeta)>dR)
			return 0;
		float deltaphi=phi_-phi;
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

private:
	float eta_,phi_;
};

class seedMaker{
public:
	seedMaker(){}

	void createSeedsFromSimClusters(const std::vector<float> *etas,
			const std::vector<float> *phis);

	const std::vector<seed>& seeds()const{return seeds_;}

	void clear(){
		seeds_.clear();
	}
private:
	std::vector<seed> seeds_;

};



#endif /* INCLUDE_SEEDMAKER_H_ */
