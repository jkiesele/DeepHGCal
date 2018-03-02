//allows functions with 18 or less paramenters
#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
//#include "boost/filesystem.hpp"
#include <iostream>
#include <stdint.h>
#include "TString.h"
#include <string>
#include <vector>
#include "TFile.h"
#include "TTree.h"
//don't use new root6 stuff, has problems with leaf_list
//#include "TTreeReader.h"
//#include "TTreeReaderValue.h"
//#include "TTreeReaderArray.h"
//#include "TTreeReaderUtils.h"
#include <boost/python/exception_translator.hpp>
#include <exception>
#include "TStopwatch.h"
#include "indata.h"
#include "pythonToSTL.h"
#include "helper.h"
#include <cmath>

using namespace boost::python; //for some reason....

static TString treename="deepntuplizer/tree";





int square_bins(
        double xval, double xcenter,
        int nbins, double half_width,
        bool isPhi, double& bincentre) {
    double bin_width = (2*half_width)/nbins;
    double low_edge = 0;
    if(isPhi)
        low_edge =deltaPhi( xcenter, half_width);
    else
        low_edge = xcenter - half_width;
    int ibin = 0;
    if(isPhi)
        ibin=std::floor((double)deltaPhi(xval,low_edge)/bin_width);
    else
        ibin=std::floor((xval - low_edge)/bin_width);

    ibin=(ibin >= 0 && ibin < nbins) ? ibin : -1;

    bincentre=(float)ibin * bin_width + low_edge + bin_width/2;
    while(isPhi && bincentre>=3.14159265358979323846)
        bincentre-=3.14159265358979323846;
    while(isPhi && bincentre<-3.14159265358979323846)
        bincentre+=3.14159265358979323846;

    return ibin;
}
bool branchIsPhi(std::string branchname){
    TString bn=branchname;
    bn.ToLower();
    return bn.Contains("phi");
}
double deltaR(const double& phi, const double& eta,
        const double& phib, const double& etab){
    double deltaphi=deltaPhi(phi,phib);
    double deltaeta=eta-etab;
    return std::sqrt(deltaphi*deltaphi+deltaeta*deltaeta);
}



/*
 * very hardcoded but likely not subject to change
 */
void fillRecHitMap_priv(boost::python::numeric::array numpyarray,std::string filename,
        int maxhitsperpixel,
        int xbins, float xwidth,int maxlayers, bool addtiming
){

    int ybins=xbins;
    float ywidth=xwidth;
    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get(treename);

    int layer_offset=1;


    std::string xbranch="rechit_phi";
    std::string ybranch="rechit_eta";
    std::string xcenter="seed_phi";
    std::string ycenter="seed_eta";
    std::string counter_branch="nrechits";

    __hidden::indata rh_energybranch;
    rh_energybranch.createFrom({"rechit_energy"}, {1.}, {0.}, MAXBRANCHLENGTH);
    __hidden::indata rh_timebranch;
    if(addtiming)
    	rh_timebranch.createFrom(  {"rechit_time"}, {1.}, {0.}, MAXBRANCHLENGTH);

    __hidden::indata layerbranch;
    layerbranch.createFrom({"rechit_layer"}, {1.}, {0.}, MAXBRANCHLENGTH);

    __hidden::indata rh_phi_eta;
    rh_phi_eta.createFrom({xbranch, ybranch}, {1., 1.}, {0., 0.}, MAXBRANCHLENGTH);

    __hidden::indata seed_phi_eta;
    seed_phi_eta.createFrom({xcenter, ycenter}, {1., 1.}, {0., 0.}, 1);

    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);


    rh_energybranch.setup(tree);
    if(addtiming)
    	rh_timebranch.setup(tree);
    layerbranch.setup(tree);
    //
    rh_phi_eta.setup(tree);
    seed_phi_eta.setup(tree);
    counter.setup(tree);

    bool rechitsarevector=rh_energybranch.isVector();

    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
    for(int it=0;it<nevents;it++){

        rh_energybranch.zeroAndGet(it);
        if(addtiming)
        	rh_timebranch.zeroAndGet(it);
        layerbranch.zeroAndGet(it);

        rh_phi_eta.zeroAndGet(it);
        seed_phi_eta.zeroAndGet(it);
        counter.zeroAndGet(it);

        std::vector<std::vector<std::vector<float> > >
        entriesperpixel(xbins,std::vector<std::vector<float> >(ybins,std::vector<float>(maxlayers,0)));

        double seedphi=seed_phi_eta.getData(0, 0);
        double seedeta=seed_phi_eta.getData(1, 0);
        int nrechits = counter.getData(0, 0);
        if(rechitsarevector){
        	nrechits = rh_energybranch.vectorSize(0);
        }

        for(size_t hit=0; hit < nrechits; hit++) {
            double bincentrephi,bincentreeta;
            double rechitphi=rh_phi_eta.getData(0, hit);
            double rechiteta=rh_phi_eta.getData(1, hit);

            int phibin = square_bins(rechitphi, seedphi, xbins, xwidth,true,bincentrephi);
            int etabin = square_bins(rechiteta, seedeta, ybins, ywidth,false,bincentreeta);


            if(phibin == -1 || etabin == -1) continue;

            int layer=0;
            layer=round(layerbranch.getData(0, hit))-layer_offset;

            if(layer>=maxlayers)
                layer=maxlayers-1;
            if(layer<0)
                layer=0;


            float drbinseed=deltaR(bincentrephi,bincentreeta,seedphi,seedeta);

            float energy=rh_energybranch.getData(0, hit);
            float time=0;
            if(addtiming)
            	time=rh_energybranch.getData(0, hit);
            float dphihitbincentre=deltaPhi(rechitphi,bincentrephi);
            float detahitbincentre=deltaPhi(rechiteta,bincentreeta);




            int offset=0;
            if(addtiming)
            	offset=entriesperpixel.at(phibin).at(etabin).at(layer)*4+2;
            else
            	offset=entriesperpixel.at(phibin).at(etabin).at(layer)*3+2;
            bool ismulti=false;
            if(entriesperpixel.at(phibin).at(etabin).at(layer)>=maxhitsperpixel){
                std::cout << phibin << ", "<< etabin << ". "<<layer<<" e "<<entriesperpixel.at(phibin).at(etabin).at(layer)<< std::endl;
                std::cout << dphihitbincentre << " - "<< detahitbincentre << std::endl;
                std::cout << "max hits per pixel reached. "<< entriesperpixel.at(phibin).at(etabin).at(layer)<<"/"
                        <<maxhitsperpixel<< ", "<<offset<< " : "<< it<< std::endl;
                if(addtiming)
                	offset-=4;
                else
                	offset-=3;
                ismulti=true;
            }

            numpyarray[it][phibin][etabin][layer][0]=drbinseed;
            numpyarray[it][phibin][etabin][layer][1]=((float)layer)/50;
            numpyarray[it][phibin][etabin][layer][offset]  +=energy;
            if(addtiming){
            	numpyarray[it][phibin][etabin][layer][offset+1]+=time;
            	numpyarray[it][phibin][etabin][layer][offset+2]+=dphihitbincentre;
            	numpyarray[it][phibin][etabin][layer][offset+3]+=detahitbincentre;
            }
            else{
            	numpyarray[it][phibin][etabin][layer][offset+1]+=dphihitbincentre;
            	numpyarray[it][phibin][etabin][layer][offset+2]+=detahitbincentre;
            }

            if(entriesperpixel.at(phibin).at(etabin).at(layer)<maxhitsperpixel){
                entriesperpixel.at(phibin).at(etabin).at(layer)+=1;
            }
        }



    }
    tfile->Close();
    delete tfile;
}

void fillRecHitMap(boost::python::numeric::array numpyarray,std::string filename,
        int maxhitsperpixel,
        int xbins, float xwidth,int maxlayers
){
	fillRecHitMap_priv(numpyarray,filename,
        maxhitsperpixel,
        xbins,  xwidth, maxlayers, true);
}
void fillRecHitMapNoTime(boost::python::numeric::array numpyarray,std::string filename,
        int maxhitsperpixel,
        int xbins, float xwidth,int maxlayers
){
	fillRecHitMap_priv(numpyarray,filename,
        maxhitsperpixel,
        xbins,  xwidth, maxlayers, false);
}

void setTreeName(std::string name){
    treename=name;
}

// Expose classes and methods to Python
BOOST_PYTHON_MODULE(c_createRecHitMap) {
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    __hidden::indata();//for some reason exposing the class prevents segfaults. garbage collector?
    //anyway, it doesn't hurt, just leave this here
    def("fillRecHitMap", &fillRecHitMap);
    def("fillRecHitMapNoTime", &fillRecHitMapNoTime);
    def("setTreeName", &setTreeName);
}
