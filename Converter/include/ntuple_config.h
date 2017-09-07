/*
 * ntuple_config.h
 *
 *  Created on: 31 Aug 2017
 *      Author: jkiesele
 */

#ifndef CONVERTER_INCLUDE_NTUPLE_CONFIG_H_
#define CONVERTER_INCLUDE_NTUPLE_CONFIG_H_

#include "ntuple_globals.h"
#include "ntuple_recHits.h"

/*
 *
 * Just a container for all content classes
 * Mostly used for merging, to have the definition only once
 *
 */
class ntuple_config{
public:
    ntuple_config(){
        all_.push_back(&gl_);
        all_.push_back(&rh_);
    }
    std::vector<ntuple_content*>& getAll(){return all_;}

private:
    ntuple_globals gl_;
    ntuple_recHits rh_;
    std::vector<ntuple_content*> all_;
};


#endif /* CONVERTER_INCLUDE_NTUPLE_CONFIG_H_ */
