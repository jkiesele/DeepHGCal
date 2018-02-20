/*
 * ntuple_config.h
 *
 *  Created on: 31 Aug 2017
 *      Author: jkiesele
 */

#ifndef CONVERTER_INCLUDE_NTUPLE_CONFIG_H_
#define CONVERTER_INCLUDE_NTUPLE_CONFIG_H_

#include "NTupleGlobals.h"
#include "NTupleRecHits.h"

/*
 *
 * Just a container for all content classes
 * Mostly used for merging, to have the definition only once
 *
 */
class NTupleConfig{
public:
    NTupleConfig(){
        all_.push_back(&gl_);
        all_.push_back(&rh_);
    }
    std::vector<NTupleContent*>& getAll(){return all_;}

private:
    NTupleGlobals gl_;
    NTupleRecHits rh_;
    std::vector<NTupleContent*> all_;
};


#endif /* CONVERTER_INCLUDE_NTUPLE_CONFIG_H_ */
