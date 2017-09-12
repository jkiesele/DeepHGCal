/*
 * truthCreator.h
 *
 *  Created on: 31 Aug 2017
 *      Author: jkiesele
 */

#ifndef CONVERTER_INCLUDE_TRUTHCREATOR_H_
#define CONVERTER_INCLUDE_TRUTHCREATOR_H_

/*
 *
 * Class to create meaningful truth information
 * E.g. if a particle decays with a narrow cone close to
 * the entry point to the calorimeter, not all its decay particles
 * are considered truth information, but the parent particle
 *
 *
 */
#include <vector>
class truthTarget{
public:

    truthTarget(float eta, float phi, float energy, float pt, int pid):
    eta_(eta),phi_(phi),energy_(energy),pt_(pt),pdgId_(pid){

    }

    float energy() const {
        return energy_;
    }

    void setEnergy(float energy) {
        energy_ = energy;
    }

    float eta() const {
        return eta_;
    }

    void setEta(float eta) {
        eta_ = eta;
    }

    int pdgId() const {
        return pdgId_;
    }

    void setPdgId(int pdgId) {
        pdgId_ = pdgId;
    }

    float phi() const {
        return phi_;
    }

    void setPhi(float phi) {
        phi_ = phi;
    }

    float pt() const {
        return pt_;
    }

    void setPt(float pt) {
        pt_ = pt;
    }

private:
    float eta_;
    float phi_;
    float energy_;
    float pt_;
    int pdgId_;
};

class truthCreator{
public:
    truthCreator():zplanecut_(0.001){}

    std::vector<truthTarget> createTruthTargets(
            const std::vector<float> *etas,
            const std::vector<float> *phis,
            const std::vector<float> *energies,
            const std::vector<float> *pts,
            const std::vector<float> *ovz,
            const std::vector<float> *dvz,
            const std::vector<int>*pids)const{
        if(zplanecut_>0)
            return createTruthFromZPlane(etas,phis,energies,pts,ovz,dvz,pids);
        else
            return std::vector<truthTarget>();//TBI
    }

private:
    float zplanecut_;

    std::vector<truthTarget> createTruthFromZPlane(
            const std::vector<float> *etas,
            const std::vector<float> *phis,
            const std::vector<float> *energies,
            const std::vector<float> *pts,
            const std::vector<float> *ovz,
            const std::vector<float> *dvz,
            const std::vector<int>*pids)const;

    /*
     * space to implement more sophisticated methods
     */

};


#endif /* CONVERTER_INCLUDE_TRUTHCREATOR_H_ */
