/*
 * ntuple_globals.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_NTUPLE_GLOBALS_H_
#define INCLUDE_NTUPLE_GLOBALS_H_

#include "ntuple_content.h"

class ntuple_globals : public ntuple_content{
public:
	ntuple_globals();

	void reset();

	void initDNNBranches(TTree* t);

	void setSeedInfo(const float& eta,
			const float& phi,
			const size_t& seedindex,const size_t& event){
		seed_eta_=eta;
		seed_phi_=phi;
		seed_index_=seedindex;
		event_=event;
	}

	void setMultiCluster(const float& eta,
            const float& phi,
            const float& energy){
        multicluster_eta_=eta;
        multicluster_phi_=phi;
        multicluster_energy_=energy;
    }

	void setTruthKinematics(const float& eta,
			const float& phi,
			const float& energy){
		true_eta_=eta;
		true_phi_=phi;
		true_energy_=energy;
	}

	void setTruthID(const int& pdgid,const bool& matched);


	//void setTruthVertex(); ///..later

private:


	//global info branches
	int event_;

	//DNN reco branches
	float seed_eta_;
	float seed_phi_;
	int   seed_index_;

	//for comparison
    float multicluster_eta_;
    float multicluster_phi_;
    float multicluster_energy_;


	//DNN truth branches
	float true_eta_;
	float true_phi_;
	float true_energy_;

	int isGamma_;
	int isHadron_;
	int isFake_;
	//to be extended

	//to be used only later
	float true_vtx_x_;
	float true_vtx_y_;
	float true_vtx_z_;



};


#endif /* INCLUDE_NTUPLE_GLOBALS_H_ */
