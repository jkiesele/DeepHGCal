/*
 * Transform.h
 *
 *  Created on: 6 Aug 2017
 *      Author: user
 */

#ifndef INCLUDE_TRANSFORMER_H_
#define INCLUDE_TRANSFORMER_H_

#include <vector>


using namespace std;

class Transformer {
public:
	Transformer(vector<float>   *rechit_x, vector<float>   *rechit_y);

	vector<float> transform(float x, float y) const;

	~Transformer() {}

	static constexpr float DX = 1.0f;
	static constexpr float DY = 1.0f;

private:
	float origin_x;
	float origin_y;
};

#endif /* INCLUDE_TRANSFORMER_H_ */
