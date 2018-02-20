//
// Created by srq2 on 2/15/18.
//

#include "iostream"
#include <math.h>
#include <TFile.h>
#include <TTree.h>
#include "TString.h"
#include "include/Converter.h"
#include "TCanvas.h"


using namespace std;


int main(int argc, char ** argv) {
    if (argc < 3) {
        cout << "USAGE:\nconvert <input file path> <output file path>" << endl;
    }

    TString infile=argv[1];
    TString outfile=argv[2];


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

    Converter cv(tree);
    cv.setOutFile(outfile);

    try{
        cv.Loop();
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