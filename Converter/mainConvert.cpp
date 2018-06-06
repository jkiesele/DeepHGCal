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
#include "boost/program_options.hpp"

namespace po = boost::program_options;


using namespace std;

int main(int argc, char ** argv) {
    po::options_description desc("Options");
        desc.add_options()
                ("help,h", "Print help")
                ("z-cut,z", po::value<float>()->default_value(0.001), "Value of the z-plane cut")
                ("no-indices,n", po::bool_switch()->default_value(false),
                 "Whether to run in no-indices mode for legacy data")
                ("energy-threshold,t", po::value<float>()->default_value(-1),
                 "Energy threshold to pick particles")
                ("have-simclusters-indices,c", po::bool_switch()->default_value(false),
                 "Energy threshold to pick particles")
                ("input,i", po::value<string>(), "Input file")
                ("output,o", po::value<string>(), "Output file");

        po::positional_options_description positional;
        positional.add("input", 1);
        positional.add("output", 2);

        po::variables_map vm;
        try {
            po::store(po::command_line_parser(
                    argc, argv).options(desc).positional(positional).run(), vm);

            if (vm.count("help")) {
                std::cout << "Converter" << std::endl
                          << desc << std::endl;
                return 0;
            }


            po::notify(vm);
        }
        catch (po::error &e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return -1;
        }
        float zCut = vm["z-cut"].as<float>();
        float energyThreshold = vm["energy-threshold"].as<float>();
        bool noIndices = vm["no-indices"].as<bool>();
        bool simClustersHaveIndices = vm["have-simclusters-indices"].as<bool>();
        TString infile = vm["input"].as<string>();
        TString outfile = vm["output"].as<string>();

        cout<<"z-cut "<< zCut<<endl;
        cout<<"no-indices "<< noIndices<<endl;
        cout<<"input "<< infile<<endl;
        cout<<"output "<< outfile<<endl;

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

        Converter cv(tree, noIndices, zCut, energyThreshold, simClustersHaveIndices);
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
