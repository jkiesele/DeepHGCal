//
// Created by Shah Rukh Qasim on 2/23/18.
//


#include "iostream"
#include <math.h>
#include <TFile.h>
#include <TTree.h>
#include "TString.h"
#include "Converter.h"
#include "TCanvas.h"
#include "IndexSimClusters.h"


using namespace std;


int main(int argc, char ** argv) {
    if (argc < 2) {
        cout << "USAGE:\nconvert <file path>" << endl;
    }

    TString infile=argv[1];


    TFile * f=new TFile(infile,"update");
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

    IndexSimClusters cv(tree);
    try{
        cv.execute();
    }catch(std::exception& e){
        f->Close();
        delete f;
        throw e;
    }

    f->Close();
    delete f;

    cout << "end conversion" << endl;
    return 0;
}