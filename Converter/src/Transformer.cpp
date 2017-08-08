/*
 * Transform.cpp
 *
 *  Created on: 6 Aug 2017
 *      Author: user
 */

#include "../include/Transformer.h"
#include <numeric>

using namespace std;

Transformer::Transformer(vector<float>   *rechit_x, vector<float>   *rechit_y) {
	// TODO Auto-generated constructor stub

	this->origin_x = accumulate( rechit_x->begin(), rechit_x->end(), 0.0) / rechit_x->size();
	this->origin_y = accumulate( rechit_y->begin(), rechit_y->end(), 0.0) / rechit_y->size();
}

vector<float> Transformer::transform(float x, float y) const
{
	//TODO DX,DY are changing according to layer. infrastructure might cause inconsistencies.
	// next step: deduce structure from data.
	float a =  (x - origin_x) / Transformer::DX;
	float b = (y - origin_y) / Transformer::DY;
	a = int(b) % 2 == 0 ? a :  a - 0.5;

	vector<float> ret = {a, b};

	return ret;
}


//Transformer::~Transformer() {
	// TODO Auto-generated destructor stub
//}
