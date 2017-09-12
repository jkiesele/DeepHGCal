/*
 * truthCreator.cpp
 *
 *  Created on: 7 Sep 2017
 *      Author: jkiesele
 */
#include "../include/truthCreator.h"
#include <stdexcept>

std::vector<truthTarget> truthCreator::createTruthFromZPlane(
        const std::vector<float> *etas,
        const std::vector<float> *phis,
        const std::vector<float> *energies,
        const std::vector<float> *pts,
        const std::vector<float> *ovz,
        const std::vector<float> *dvz,
        const std::vector<int>*pids)const{
    if(etas->size()!=phis->size() || etas->size()!=energies->size() ||
            etas->size()!=pts->size() || etas->size()!=pids->size() )
        throw std::out_of_range("truthCreator::createTruthFromZPlane: inputs must be same length");

    std::vector<truthTarget> out;

    for(size_t i=0;i<etas->size();i++){
        const float direction= etas->at(i)>=0 ? 1. : -1.;
        if(direction*ovz->at(i) >= zplanecut_) continue;
        if(direction*dvz->at(i) <  zplanecut_) continue;

        out.push_back(truthTarget(
                etas->at(i),
                phis->at(i),
                energies->at(i),
                pts->at(i),
                pids->at(i)));

    }

    return out;
}
