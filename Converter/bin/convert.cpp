/*
 * convert.cpp
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */


#include <iostream>
#include "TString.h"

#include "../include/converter.h"
#include "TCanvas.h"
#include <iostream>

int main(int argc, char *argv[]){

	if(argc<3){
		std::cout << "USAGE:\nconvert <input file path> <output file path> <optional: test>"<<std::endl;
	}

	TString infile=argv[1];
	TString outfile=argv[2];

	TString extra="";
	if(argc>3){
	    extra=argv[3];
	}

	//open infile
	TFile * f=new TFile(infile,"READ");
	if(!f || f->IsZombie()){
		std::cerr << "Input file not found" <<std::endl;
		if(f->IsZombie())
			delete f;
		return -1;
	}

	TTree* tree= (TTree*)f->Get("ana/hgc");

	if(!tree || tree->IsZombie()){
		std::cerr << "Input tree not found" <<std::endl;
		f->Close();
		delete f;
		return -2;
	}

	converter cv(tree);
	cv.setOutFile(outfile);
	if(extra.Contains("test")){
	    cv.setTest(true);
	    std::cout << "Test mode: only running on 50 events" <<std::endl;
	}

	try{
		cv.Loop();
	}catch(std::exception& e){
		f->Close();
		delete f;
		throw e;
	}



	f->Close();
	delete f;
	return 0;
}

