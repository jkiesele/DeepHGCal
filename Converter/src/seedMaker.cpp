/*
 * seedMaker.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */



#include "../include/seedMaker.h"
#include <stdexcept>

void seedMaker::createSeedsFromCollection(const std::vector<float> *etas,
        const std::vector<float> *phis,
        const std::vector<bool> * select){

    if(etas->size()!=phis->size())
        throw std::out_of_range("seedMaker::createSeedsFromSimClusters: eta and phi vectors not of same size");

    for(size_t i=0;i<etas->size();i++){
        if(select){
            if(! select->at(i)) continue;
        }
        if(fabs(etas->at(i))>maxeta || fabs(etas->at(i))<mineta)
            continue;
        seed newseed(etas->at(i),phis->at(i));
        seeds_.push_back(newseed);
    }
}
void seedMaker::createSeedsFromCollection(const std::vector<float> *etas,
        const std::vector<float> *phis,
        const std::vector<float> * select,
        const float selectcut){

    if(etas->size()!=phis->size())
        throw std::out_of_range("seedMaker::createSeedsFromSimClusters: eta and phi vectors not of same size");

    for(size_t i=0;i<etas->size();i++){
        if(select){
            if(select->at(i) < selectcut) continue;
        }
        if(fabs(etas->at(i))>maxeta || fabs(etas->at(i))<mineta)
            continue;
        seed newseed(etas->at(i),phis->at(i));
        seeds_.push_back(newseed);
    }


}
