/*
 * seedMaker.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */



#include "../include/seedMaker.h"
#include <stdexcept>

void seedMaker::createSeedsFromSimClusters(const std::vector<float> *etas,
		const std::vector<float> *phis){

	if(etas->size()!=phis->size())
		throw std::out_of_range("seedMaker::createSeedsFromSimClusters: eta and phi vectors not of same size");

	for(size_t i=0;i<etas->size();i++){
		seed newseed(etas->at(i),phis->at(i));
		seeds_.push_back(newseed);
	}


}
