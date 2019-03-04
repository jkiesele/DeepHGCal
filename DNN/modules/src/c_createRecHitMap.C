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
#include "TRandom3.h"
#include "TChain.h"
#include <algorithm>
#include <random>

using namespace boost::python; //for some reason....

static TString treename="deepntuplizer/tree";

template<typename _RandomAccessIterator, typename _Compare>
inline std::vector<size_t>
retsort(_RandomAccessIterator __first, _RandomAccessIterator __last,
		_Compare __comp){
	typedef typename std::iterator_traits<_RandomAccessIterator>::value_type
			_ValueType;
	std::vector<_ValueType> copy(__first,__last); //copy
	std::vector<size_t> sortedilo;
	std::sort(copy.begin(),copy.end(),__comp);

	for(_RandomAccessIterator it=copy.begin();it!=copy.end();++it){
			//get the position in input
			size_t pos=std::find(__first,__last,*it)-__first;
			sortedilo.push_back(pos);
	}
	std::copy(copy.begin(),copy.end(),__first);
	return sortedilo;
}




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
        int xbins, float xwidth,
		int ybins, float ywidth,
		int maxlayer, int minlayer, bool addtiming
){

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
        entriesperpixel(xbins,std::vector<std::vector<float> >(ybins,std::vector<float>(maxlayer,0)));

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

            if(layer>=maxlayer)
                continue;
            if(layer<minlayer)
                continue;


            float drbinseed=deltaR(bincentrephi,bincentreeta,seedphi,seedeta);

            float energy=rh_energybranch.getData(0, hit);
            float time=0;
            if(addtiming)
            	time=rh_timebranch.getData(0, hit);
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


void simple3Dstructure(boost::python::numeric::array numpyarray,std::string filename,
        int xbins, float xwidth,
		int ybins, float ywidth,
		int maxlayer, int minlayer,
		bool sumenergy
){


	//make this a chain
    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get(treename);



    int layer_offset=0;

    int maxhitsperpixel=1;

    std::string xbranch="rechit_x";
    std::string ybranch="rechit_y";
    std::string counter_branch="nrechits";

    __hidden::indata rh_energybranch;
    rh_energybranch.createFrom({"rechit_energy"}, {1.}, {0.}, MAXBRANCHLENGTH);
    __hidden::indata rh_timebranch;

    __hidden::indata layerbranch;
    layerbranch.createFrom({"rechit_layer"}, {1.}, {0.}, MAXBRANCHLENGTH);

    __hidden::indata rh_phi_eta;
    rh_phi_eta.createFrom({xbranch, ybranch}, {1., 1.}, {0., 0.}, MAXBRANCHLENGTH);

    __hidden::indata counter;
    counter.createFrom({counter_branch}, {1.}, {0.}, 1);


    rh_energybranch.setup(tree);
    layerbranch.setup(tree);
    //
    rh_phi_eta.setup(tree);
    counter.setup(tree);

    bool rechitsarevector=rh_energybranch.isVector();

    const int nevents=std::min( (int) tree->GetEntries(), (int) boost::python::len(numpyarray));
    for(int it=0;it<nevents;it++){

        rh_energybranch.zeroAndGet(it);
        layerbranch.zeroAndGet(it);

        rh_phi_eta.zeroAndGet(it);
        counter.zeroAndGet(it);

        std::vector<std::vector<std::vector<float> > >
        entriesperpixel(xbins,std::vector<std::vector<float> >(ybins,std::vector<float>(maxlayer-minlayer,0)));

        double seedphi=0;//seed_phi_eta.getData(0, 0);
        double seedeta=0;//seed_phi_eta.getData(1, 0);
        int nrechits = counter.getData(0, 0);
        if(rechitsarevector){
        	nrechits = rh_energybranch.vectorSize(0);
        }


        double newphi=0; //phisum/energysum;
        double neweta=0; //etasum/energysum;

        seedphi = seedphi+newphi;
        if(seedphi>3.14159265358979323846)seedphi-=3.14159265358979323846;
        if(seedphi<3.14159265358979323846)seedphi+=3.14159265358979323846;
        seedeta = neweta;


        for(size_t hit=0; hit < nrechits; hit++) {
            double bincentrephi,bincentreeta;
            double rechitphi=rh_phi_eta.getData(0, hit);
            double rechiteta=rh_phi_eta.getData(1, hit);

            int phibin = square_bins(rechitphi, seedphi, xbins, xwidth,true,bincentrephi);
            int etabin = square_bins(rechiteta, seedeta-0.0001, ybins, ywidth,false,bincentreeta);


            if(phibin == -1 || etabin == -1) continue;

            int layer=0;
            layer=round(layerbranch.getData(0, hit))-layer_offset;

            if(layer>=maxlayer)
                continue;
            if(layer<minlayer)
                continue;
            layer-=minlayer;

            float drbinseed=deltaR(bincentrephi,bincentreeta,seedphi,seedeta);

            float energy=1000*rh_energybranch.getData(0, hit);
            float dphihitbincentre=deltaPhi(rechitphi,bincentrephi);
            float detahitbincentre=deltaPhi(rechiteta,bincentreeta);


            //std::cout << phibin << ", " << etabin << ": " << energy << std::endl;


            bool ismulti=false;
            if(!sumenergy && entriesperpixel.at(phibin).at(etabin).at(layer)>=maxhitsperpixel){
                std::cout << phibin << ", "<< etabin << ". "<<layer<<" e "<<entriesperpixel.at(phibin).at(etabin).at(layer)<< std::endl;
                std::cout << dphihitbincentre << " - "<< detahitbincentre << std::endl;
                std::cout << "max hits per pixel reached. "<< entriesperpixel.at(phibin).at(etabin).at(layer)<<"/"
                        <<maxhitsperpixel<< ", "<< " : "<< it<< std::endl;

                ismulti=true;
            }

            //numpyarray[it][phibin][etabin][layer][0]=drbinseed;
            //numpyarray[it][phibin][etabin][layer][0]=layer+minlayer;
            numpyarray[it][phibin][etabin][layer][0]  +=energy;


            if(entriesperpixel.at(phibin).at(etabin).at(layer)<maxhitsperpixel){
                entriesperpixel.at(phibin).at(etabin).at(layer)+=1;
            }
        }



    }
    tfile->Close();
    delete tfile;
}

void checkBranchSetReturn(int ret, TString branchname_ ){
	if(ret == -2 || ret == -1){
		std::cout << "tBranchHandler: Class type given for branch " << branchname_
				<< " does not match class type in tree. (root CheckBranchAddressType returned " << ret << ")" <<std::endl;
		throw std::runtime_error("tBranchHandler: Class type does not match class type in branch");
	}
	else if(ret == -4 || ret == -3){
		std::cout << "tBranchHandler: Internal error in branch " << branchname_
				<< " (root CheckBranchAddressType returned " << ret << ")" <<std::endl;
		throw std::runtime_error("tBranchHandler: Internal error in branch");
	}
	else if( ret == -5){
		std::cout << "tBranchHandler: branch " << branchname_ << " does not exists!" << std::endl;
		throw std::runtime_error("tBranchHandler: branch does not exists!");

	}
}
void simpleRandom3Dstructure(boost::python::numeric::array numpyarray,
		const boost::python::list infiles,
        int xbins, float xwidth,
		int ybins, float ywidth,
		int maxlayer, int minlayer,
		int seed
){

	const bool sumenergy=true;
	std::vector<TString>  allfiles = toSTLVector<TString>(infiles);
	const double seedeta=0.36;
    const int noutputevents=boost::python::len(numpyarray);
    int doneoutputevents=0;

	int layer_offset=0;
	int branchset=0;



	TRandom3 rand(seed);

	std::cout << "PU_mixer: adding PU rechits.." << std::endl;

	int readfile=(int)rand.Uniform(0,allfiles.size()-1);
    size_t addedfiles=0;
    while(doneoutputevents<noutputevents){

    	if(readfile>=allfiles.size())
    		readfile=0;

    	TFile * tfile=new TFile(allfiles.at(readfile), "READ");
    	TTree * tree = (TTree*) tfile->Get(treename);
    	if(!tree || tree->IsZombie()){
    		delete tfile;
    		continue;
    	}
    	std::cout << "PU_mixer: using file " << allfiles.at(readfile) << std::endl;
    	readfile++;

    	std::vector<int> * rechit_layer=0;
    	auto rechit_layer_branch=tree->GetBranch("rechit_layer");
    	rechit_layer_branch->SetAddress(&rechit_layer);

    	std::vector<float> * rechit_eta=0;
    	auto rechit_eta_branch=tree->GetBranch("rechit_eta");
    	rechit_eta_branch->SetAddress(&rechit_eta);

    	std::vector<float> * rechit_phi=0;
    	auto rechit_phi_branch=tree->GetBranch("rechit_phi");
    	rechit_phi_branch->SetAddress(&rechit_phi);

    	std::vector<float> * rechit_energy=0;
    	auto rechit_energy_branch=tree->GetBranch("rechit_energy");
    	rechit_energy_branch->SetAddress(&rechit_energy);


    	tree->SetBranchStatus("*",0);  // disable all branches
    	tree->SetBranchStatus("rechit_eta",1);
    	tree->SetBranchStatus("rechit_phi",1);
    	tree->SetBranchStatus("rechit_layer",1);
    	tree->SetBranchStatus("rechit_energy",1);


    	const int ntreeevents=tree->GetEntries();

    	int it=0;//(int)rand.Uniform(0,ntreeevents-1);//starting event

    	// event loop
    	for(int it=0;it<ntreeevents;it++){
    		if(doneoutputevents>=noutputevents) break;
    		tree->GetEntry(it);

    		size_t nrechits = rechit_layer->size();//rechit_layer.size();

    		if(!nrechits || nrechits>1e9)continue;

    		const float signs[]={-1,1};

    		double seedphi=2.*3.1415926536/704.*(int)rand.Uniform(0,704.5);

    		//use the same event a few times first to reduce IO

    		for(size_t etasignselect=0;etasignselect<2;etasignselect++){
    			double etamult=signs[etasignselect];
    			if(doneoutputevents>=noutputevents) break;
    			for(float phiadd=0;phiadd<2.*3.1415926536;phiadd+=0.2){
    				if(doneoutputevents>=noutputevents) break;
    				seedphi+=phiadd;
    				if(seedphi>=2.*3.1415926536)seedphi-=2.*3.1415926536;

    				for(size_t hit=0; hit < nrechits; hit++) {

    					int layer=rechit_layer->at(hit);
    					if(layer>=maxlayer)
    						continue;
    					if(layer<minlayer)
    						continue;
    					layer-=minlayer;

    					double bincentrephi,bincentreeta;
    					const float& rechitphi=rechit_phi->at(hit);
    					if(fabs(deltaPhi(rechitphi,seedphi))>xwidth) continue;

    					const double rechiteta=etamult*rechit_eta->at(hit);
    					if(fabs(seedeta-rechiteta)>ywidth)continue;

    					int phibin = square_bins(rechitphi, seedphi, xbins, xwidth,true,bincentrephi);
    					int etabin = square_bins(rechiteta, seedeta-0.0001, ybins, ywidth,false,bincentreeta);


    					if(phibin == -1 || etabin == -1) continue;


    					float drbinseed=deltaR(bincentrephi,bincentreeta,seedphi,seedeta);

    					float energy=1000*rechit_energy->at(hit);
    					float dphihitbincentre=deltaPhi(rechitphi,bincentrephi);
    					float detahitbincentre=deltaPhi(rechiteta,bincentreeta);

    					//numpyarray[doneoutputevents][phibin][etabin][layer][0]=layer+minlayer;
    					numpyarray[doneoutputevents][phibin][etabin][layer][0]  +=energy;

    				}
    				//std::cout <<  "PU_mixer: added PU event: "<< doneoutputevents<<"/" <<noutputevents<<std::endl;
    				doneoutputevents++;
    			}
    		}
    	}
    	if(rechit_layer)
    		delete rechit_layer, rechit_energy, rechit_eta, rechit_phi;
    	delete tfile;
    }
    //delete tree;
}

void fillRecHitMap(boost::python::numeric::array numpyarray,std::string filename,
        int maxhitsperpixel,
        int xbins, float xwidth,
		int ybins, float ywidth,
		int maxlayers, int minlayer
){
	fillRecHitMap_priv(numpyarray,filename,
        maxhitsperpixel,
        xbins,  xwidth,
		ybins, ywidth,
		maxlayers,minlayer, true);
}
void fillRecHitMapNoTime(boost::python::numeric::array numpyarray,std::string filename,
        int maxhitsperpixel,
        int xbins, float xwidth,
		int ybins, float ywidth,
		int maxlayer, int minlayer
){
	fillRecHitMap_priv(numpyarray,filename,
        maxhitsperpixel,
        xbins,  xwidth,
		ybins, ywidth, maxlayer, minlayer, false);
}


//////////////

void fillRecHitList_priv(boost::python::numeric::array numpyarray,std::string filename,
        int maxrechitsperevent,float maxdr,int maxlayers, bool addtiming
){

    TFile* tfile= new TFile(filename.c_str(), "READ");
    TTree* tree = (TTree*) tfile->Get(treename);



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


        double seedphi=seed_phi_eta.getData(0, 0);
        double seedeta=seed_phi_eta.getData(1, 0);
        int nrechits = counter.getData(0, 0);
        if(rechitsarevector){
        	nrechits = rh_energybranch.vectorSize(0);
        }
        //create energy vector
        std::vector<float> energies(nrechits,0);
        for(size_t hit=0; hit < nrechits; hit++) {
        	energies[hit]=rh_energybranch.getData(0, hit);
        }
        std::vector<size_t> sortIDs=retsort(energies.begin(),energies.end(),std::greater<float>());
        for(size_t i=0; i < nrechits; i++) {
        	size_t hit= sortIDs.at(i);

            float layer=layerbranch.getData(0, hit);
            if(layer>(float)maxlayers) continue;
            if(i>=maxrechitsperevent) break;

            double rechitphi=rh_phi_eta.getData(0, hit);
            double rechiteta=rh_phi_eta.getData(1, hit);

            float dphihitseed=deltaPhi(rechitphi,seedphi);
            float detahitseed=rechiteta - seedeta;

            float energy=rh_energybranch.getData(0, hit);


            numpyarray[it][i][0]=energy;
            numpyarray[it][i][1]=dphihitseed;
            numpyarray[it][i][2]=detahitseed;
            numpyarray[it][i][3]=layer;
            if(addtiming)
            	numpyarray[it][i][4]=rh_energybranch.getData(0, hit);


        }



    }
    tfile->Close();
    delete tfile;
}

void fillRecHitList(boost::python::numeric::array numpyarray,std::string filename,
        int maxrechitsperevent,float maxdr,int maxlayers){
	fillRecHitList_priv(numpyarray, filename,
	         maxrechitsperevent, maxdr, maxlayers, true);
}

void fillRecHitListNoTime(boost::python::numeric::array numpyarray,std::string filename,
        int maxrechitsperevent,float maxdr,int maxlayers){
	fillRecHitList_priv(numpyarray, filename,
	         maxrechitsperevent, maxdr, maxlayers, false);
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
    def("fillRecHitList", &fillRecHitList);
    def("fillRecHitListNoTime", &fillRecHitListNoTime);
    def("setTreeName", &setTreeName);
    def("simple3Dstructure", &simple3Dstructure);
    def("simpleRandom3Dstructure", &simpleRandom3Dstructure);
}
