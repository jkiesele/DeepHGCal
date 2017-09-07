/*
 * Transform.cpp
 *
 *  Created on: 6 Aug 2017
 *      Author: user
 */

#include "../include/Transformer.h"
#include <numeric>
#include <map>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <string>
#include <limits>

#define LAYER_NUM 52

using namespace std;

Transformer::Transformer(vector<float> *rechit_x, vector<float> *rechit_y, vector<int> *rechit_layer, converter* c)
{
    _DX_per_layer = this->find_dx(rechit_x, c);
    _DY_per_layer = this->find_dx(rechit_y, c);

	//each value is the origins coordinate in the respective layer and axis
	_origin_x = {};
	_origin_y = {};

    for (int layer = 0; layer < LAYER_NUM; layer++)
    {
    	_origin_x.push_back(get_layer_origin(rechit_x, rechit_layer, layer));
    	_origin_y.push_back(get_layer_origin(rechit_y, rechit_layer, layer));
    }
}

float Transformer::get_layer_origin(vector<float> *rechit, vector<int> *rechit_layer, int layer)
{
	vector<float> rechits_in_layer = {};


	for (unsigned int i = 0; i < rechit->size(); i++)
	{
		if(rechit_layer->at(i) == layer)
			rechits_in_layer.push_back(rechit->at(i));
	}


	float origin = accumulate(rechits_in_layer.begin(), rechits_in_layer.end(), 0.0) / rechits_in_layer.size();

    return origin;
}

vector<float> Transformer::transform(float x, float y, int layer ) const
{

	float a = round((x - _origin_x[layer]) / _DX_per_layer[layer]);
	float b = round((y - _origin_y[layer]) / _DY_per_layer[layer]);

	a = int(b) % 2 == 0 ? a :  a - _DX_per_layer[layer];

	vector<float> ret = {a, b};

	return ret;
}


//Transformer::~Transformer() {
	// TODO Auto-generated destructor stub
//}


vector<float> Transformer::find_dx(vector<float>* rechit, converter* c)
{
	//vector of hexagon unit width per layer
    vector<float> dx = {};


    //maps hits to corresponding layer
    map<int, vector<float>> hits_by_layer;

    Long64_t nentries = c->fChain->GetEntries();
    Long64_t nb;

    //copies hits to their matching entry in the map
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
        Long64_t ientry = c->LoadTree(jentry);
        if (ientry < 0) break;
        nb = c->fChain->GetEntry(jentry);

        for(size_t i = 0; i < rechit->size(); i++){
        	hits_by_layer[c->rechit_layer->at(i)].push_back(rechit->at(i));
        }
    }

    //for each layer. find hexagon characteristic length
    for (int i = 0; i < LAYER_NUM; i++){

    	if(hits_by_layer.find(i) != hits_by_layer.end())
    	{
    		vector<float> layer = hits_by_layer[i];

    		sort(layer.begin(), layer.end());

			int diff = numeric_limits<float>::max();
			int tmp_diff = 0;

			for(unsigned int i = 1; i < layer.size(); i++){
				tmp_diff = layer[i] - layer[i-1];

				if (tmp_diff <= diff && tmp_diff > 0){
						diff = tmp_diff;
    		    }

				dx.push_back(diff);
			}
    	}
    }

    return dx;
}