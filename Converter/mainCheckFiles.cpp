

/*
 * checkFileList.cc
 *
 *  Created on: 22 Apr 2017
 *      Author: jkiesele
 */




#include "TFile.h"
#include <fstream>
#include "TString.h"
#include <iostream>
#include "TTree.h"
#include <libgen.h>

//move to a shared helper later
std::vector<TString> readSampleFile(const TString& file, const TString& addpath){
    std::string line;
    std::vector<TString>  out;
    std::ifstream myfile (file.Data());
    if (myfile.is_open()){
        while ( getline (myfile,line) ){
            if(addpath.Length())
                out.push_back(addpath+"/"+(TString)line);
            else
                out.push_back((TString)line);
        }

        myfile.close();
    }
    else
        throw std::runtime_error("could not read file");
    return out;
}


int main(int argc, char *argv[]){

    if(argc<3){
        std::cout << "USAGE: checkFileList <root file list> <tree name>\n will overwrite the existing\
 list but copy it to a backup before. The output file list will only contain good files" <<std::endl;
    }


    const TString file=argv[1];

    TString inpath=dirname(argv[1]);

    TString treename=argv[2];


    std::cout << inpath << std::endl;

    std::vector<TString>  samples=readSampleFile(file,inpath);
    std::vector<TString>  sample_filenames=readSampleFile(file,"");

    system(("mv "+file+" "+file+".backup").Data());

    std::ofstream outlist(file.Data());

    std::cout << "started checking, can take a minute" <<std::endl;

    for(size_t i=0;i<samples.size();i++){
        TFile * f = new TFile(samples.at(i));

        //maybe mor elater
        bool fileisgood = f && !f->IsZombie();

        std::cout << "checking "<< samples.at(i) << "..." ;

        if(fileisgood){

        	TTree * t = (TTree*) f->Get(treename);

        	fileisgood &= t && !t->IsZombie();

        }
        if(fileisgood){
            outlist << sample_filenames.at(i) << '\n';
            std::cout << " ok" << std::endl;
        }
        else
            std::cout << " BROKEN, removing " <<std::endl;


        f->Close();
        delete f;
    }

    outlist.close();
}
