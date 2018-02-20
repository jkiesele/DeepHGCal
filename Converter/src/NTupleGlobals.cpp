/*
 * ntuple_globals.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */




#include <iostream>
#include "../include/NTupleGlobals.h"

NTupleGlobals::NTupleGlobals():
NTupleContent()
{reset();}


void NTupleGlobals::reset(){
	event_=0;

	seed_eta_=0;
	seed_phi_=0;
	seed_index_=0;

    totalrechit_energy_=0;
    true_energyfraction_=0;

    multicluster_eta_=0;
    multicluster_phi_=0;
    multicluster_energy_=0;

    simcluster_eta_=0;
    simcluster_phi_=0;
    simcluster_energy_=0;

    true_ncloseparticles_=0;
    true_drclosestparticle_=-1;
    true_averagedr_=0;
    true_closeparticledr_.clear();

	true_eta_=0;
	true_phi_=0;
	true_pt_=0;
	true_energy_=0;
	true_pid_=0;

    isGamma_=0;
    isElectron_=0;
    isMuon_=0;
    isTau_=0;
    isPionZero_=0;
    isPionCharged_=0;
    isEta_=0;
    isProton_=0;
    isKaonCharged_=0;
    isOther_=0;
    isFake_=0;

	true_vtx_x_=0;
	true_vtx_y_=0;
	true_vtx_z_=0;

	nmulticlusters_=0;

    num_clusters_ = 0;
    num_particles_ = 0;
}

void NTupleGlobals::setTruthID(const int& pdgid,const bool& matched){

    isGamma_=0;
    isElectron_=0;
    isMuon_=0;
    isTau_=0;
    isPionZero_=0;
    isPionCharged_=0;
    isEta_=0;
    isOther_=0;
    isFake_=0;
    true_pid_=0;
    isProton_=0;
    isKaonCharged_=0;

	if(!matched){
		isFake_=1;
		return;
	}
	true_pid_=pdgid;
	//some translation here - just example for now
	if(pdgid==22)
		isGamma_=1;
	else if(std::abs(pdgid)==11)
		isElectron_=1;
    else if(std::abs(pdgid)==13)
        isMuon_=1;
    else if(std::abs(pdgid)==15)
        isTau_=1;
    else if(std::abs(pdgid)==111)
        isPionZero_=1;
    else if(std::abs(pdgid)==211)
        isPionCharged_=1;
    else if(std::abs(pdgid)==221)
        isEta_=1;
    else if(pdgid==2212)
        isProton_=1;
    else if(std::abs(pdgid)==321)
        isKaonCharged_=1;
    else
        isOther_=1;

}



void NTupleGlobals::initDNNBranches(TTree* t){


	addBranch(t,"event", &event_, "event_/i");

	addBranch(t,"seed_eta",   &seed_eta_,   "seed_eta_/f");
	addBranch(t,"seed_phi",   &seed_phi_,   "seed_phi_/f");
	addBranch(t,"seed_index", &seed_index_, "seed_index_/i");

	addBranch(t,"totalrechit_energy",   &totalrechit_energy_,   "totalrechit_energy_/f");
    addBranch(t,"true_energyfraction",   &true_energyfraction_,   "true_energyfraction_/f");

	addBranch(t,"multicluster_eta",   &multicluster_eta_,   "multicluster_eta_/f");
	addBranch(t,"multicluster_phi",   &multicluster_phi_,   "multicluster_phi_/f");
	addBranch(t,"multicluster_energy",   &multicluster_energy_,   "multicluster_energy_/f");

    addBranch(t,"simcluster_eta",      &simcluster_eta_,      "simcluster_eta_/f");
    addBranch(t,"simcluster_phi",      &simcluster_phi_,      "simcluster_phi_/f");
    addBranch(t,"simcluster_energy",   &simcluster_energy_,   "simcluster_energy_/f");


    addBranch(t,"true_ncloseparticles",   &true_ncloseparticles_,   "true_ncloseparticles_");
    addBranch(t,"true_drclosestparticle",   &true_drclosestparticle_,   "true_drclosestparticle_/f");
    addBranch(t,"true_averagedr",   &true_averagedr_,   "true_averagedr_/f");
    addBranch(t,"true_closeparticledr",   &true_closeparticledr_);


	addBranch(t,"true_eta",   &true_eta_,   "true_eta_/f");
	addBranch(t,"true_phi",   &true_phi_,   "true_phi_/f");
    addBranch(t,"true_pt",   &true_pt_,   "true_pt_/f");
	addBranch(t,"true_energy",&true_energy_,"true_energy_/f");
	addBranch(t,"true_pid",&true_pid_,"true_pid_/i");


	addBranch(t,"isGamma",   &isGamma_,   "isGamma_/i");
    addBranch(t,"isElectron",   &isElectron_,   "isElectron_/i");
    addBranch(t,"isMuon",   &isMuon_,   "isMuon_/i");
    addBranch(t,"isTau",   &isTau_,   "isTau_/i");
    addBranch(t,"isPionZero",   &isPionZero_,   "isPionZero_/i");
    addBranch(t,"isPionCharged",   &isPionCharged_,   "isPionCharged_/i");
    addBranch(t,"isProton",   &isProton_,   "isProton_/i");
    addBranch(t,"isKaonCharged",   &isKaonCharged_,   "isKaonCharged_/i");
    addBranch(t,"isEta",   &isEta_,   "isEta_/i");
    addBranch(t,"isOther",   &isOther_,   "isOther_/i");
	addBranch(t,"isFake",    &isFake_,    "isFake_/i");

	addBranch(t,"true_vtx_x",   &true_vtx_x_,   "true_vtx_x_/f");
	addBranch(t,"true_vtx_y",   &true_vtx_y_,   "true_vtx_y_/f");
	addBranch(t,"true_vtx_z",   &true_vtx_z_,   "true_vtx_z_/f");


    addBranch(t,"num_clusters",   &num_clusters_,   "num_clusters_/i");
    addBranch(t,"cluster_id",   &cluster_id_,   "cluster_id_[num_clusters_]/i");
    addBranch(t,"cluster_particle_eta",   &cluster_particle_eta_,   "cluster_particle_eta_[num_clusters_]/f");
    addBranch(t,"cluster_particle_phi",   &cluster_particle_phi_,   "cluster_particle_phi_[num_clusters_]/f");


    addBranch(t,"num_particles",   &num_particles_,   "num_particles_/i");
    addBranch(t,"particle_eta",   &particle_eta_,   "particle_eta_[num_particles_]/f");
    addBranch(t,"particle_phi",   &particle_phi_,   "particle_phi_[num_particles_]/f");
    addBranch(t,"particle_r_origin",   &particle_r_origin_,   "particle_r_origin_[num_particles_]/f");
    addBranch(t,"particle_r_decay",   &particle_r_decay_,   "particle_r_decay_[num_particles_]/f");

    addBranch(t,"particle_x_origin",   &particle_x_origin_,   "particle_x_origin_[num_particles_]/f");
    addBranch(t,"particle_y_origin",   &particle_y_origin_,   "particle_y_origin_[num_particles_]/f");
    addBranch(t,"particle_z_origin",   &particle_z_origin_,   "particle_z_origin_[num_particles_]/f");
    addBranch(t,"particle_x_decay",   &particle_x_decay_,   "particle_x_decay_[num_particles_]/f");
    addBranch(t,"particle_y_decay",   &particle_y_decay_,   "particle_y_decay_[num_particles_]/f");
    addBranch(t,"particle_z_decay",   &particle_z_decay_,   "particle_z_decay_[num_particles_]/f");


}



void NTupleGlobals::addSimClusterData(const int &id, const float &cluster_particle_eta,
                                      const float &cluster_particle_phi) {
    if (num_clusters_ > MAX_CLUSTERS) {
        std::cerr<<"Max clusters exceeding"<<std::endl;
        return;
    }
    cluster_id_[num_clusters_] = id;
    cluster_particle_eta_[num_clusters_] = cluster_particle_eta;
    cluster_particle_phi_[num_clusters_] = cluster_particle_phi;

    num_clusters_ ++;
}

void NTupleGlobals::addParticleData(const float &particleEta, const float &particlePhi, const float &particleOriginR,
                                    const float &particleDecayR,
                                    const float& particleOriginX,
                                    const float& particleOriginY,
                                    const float& particleOriginZ,
                                    const float& particleDecayX,
                                    const float& particleDecayY,
                                    const float& particleDecayZ) {

    if (num_particles_ > MAX_PARTICLES) {
        std::cerr<<"Max particles exceeding"<<std::endl;
        return;
    }

    particle_eta_[num_particles_] = particleEta;
    particle_phi_[num_particles_] = particlePhi;
    particle_r_origin_[num_particles_] = particleOriginR;
    particle_r_decay_[num_particles_] = particleDecayR;

    particle_x_origin_[num_particles_] = particleOriginX;
    particle_y_origin_[num_particles_] = particleOriginY;
    particle_z_origin_[num_particles_] = particleOriginZ;

    particle_x_decay_[num_particles_] = particleDecayX;
    particle_y_decay_[num_particles_] = particleDecayY;
    particle_z_decay_[num_particles_] = particleDecayZ;

    num_particles_ ++;
}