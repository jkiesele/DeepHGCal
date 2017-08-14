/*
 * Transform.h
 *
 *  Created on: 6 Aug 2017
 *      Author: user
 */

#ifndef INCLUDE_TRANSFORMER_H_
#define INCLUDE_TRANSFORMER_H_


#include "../include/converter.h"
#include <vector>


using namespace std;

class Transformer {
public:
	Transformer(vector<float> *rechit_x, vector<float> *rechit_y, vector<int> *rechit_layer, converter* c);

	vector<float> transform(float x, float y, int layer) const;

	~Transformer() {}

private:
	vector<float> _origin_x;
	vector<float> _origin_y;
	vector<float> _DX_per_layer;
	vector<float> _DY_per_layer;

	vector<float> find_dx(vector<float>* rechit, converter*c);
	float get_layer_origin(vector<float> *rechit,vector<int> *rechit_layer, int layer);
};

#endif /* INCLUDE_TRANSFORMER_H_ */
