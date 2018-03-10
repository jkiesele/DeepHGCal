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
#include "NTupleGlobals.h"
#include <unordered_map>

#include <vector>

using namespace std;

struct RecHitData {
	unsigned int id;
	float eta;
	float phi;
	float energy;
};

class Converter: public HGCalSel{
private:
    Long64_t loadEvent(const Long64_t &eventNo);
public:
	Converter(TTree* t, bool noIndices = false);

	void setEnergyThreshold(float thr){energylowercut_=thr;}

	void setTest(bool test){testmode_=test;}

	void setOutFile(const TString& outname){
		outfilename_=outname;
	}


    void initializeBranches();
	void Loop();

    void recomputeSimClusterEtaPhi();
    void addParticleDataToGlobals(NTupleGlobals& globals, size_t index);
    unordered_map<int, int> findSimClusterForSeeds(vector<int>& seeds);
	void traceDecayTree(unordered_set<int> &decayParticlesCluster, unordered_set<int> &allParticles) const;
        std::vector<std::unordered_map<unsigned int, float>> indexSimClusterRecHits();
    unordered_map<int, pair<vector<int>, vector<int>>> findParticlesFromCollision() const;

	TString outfilename_;

private:
	const float THRESHOLD_ASSIGN_TO_CLUSTER = 0.00001;
	bool testmode_;
	float energylowercut_;
    const float ZPLANE_CUT = 0.001;
    const float ZPLANE_CUT_CALORIMETER = 319.9;
    bool noIndices;

	std::vector<std::vector<int> > simcluster_hits_indices_computed;
};



#endif /* INCLUDE_CONVERTER_H_ */
