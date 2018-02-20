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
public:
	Converter(TTree* t):HGCalSel(t),testmode_(false),energylowercut_(0){}

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
    pair<int, float> findParticleForRecHit(int recHitId, unordered_map<int, int> &simClustersForSeeds,
                                           std::vector<std::unordered_map<unsigned int, float>> &recHitsForSimClusters);
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
};



#endif /* INCLUDE_CONVERTER_H_ */
