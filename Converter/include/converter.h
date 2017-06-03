/*
 * converter.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_CONVERTER_H_
#define INCLUDE_CONVERTER_H_

/*
 *
 * Simple wrapper to keep the 'makeClass' part
 * as clean as possible (in case of updates)
 *
 */

#include "HGCalSel.h"
#include "TTree.h"

class converter: public HGCalSel{
public:
	converter(TTree* t):HGCalSel(t){}

	void setOutFile(const TString& outname){
		outfilename_=outname;
	}

	void Loop();//overwrite standard Selector Loop

private:
	TString outfilename_;

};



#endif /* INCLUDE_CONVERTER_H_ */
