/*
 * Transformer2.h
 *
 *  Created on: 12 Sep 2017
 *      Author: user
 */

#ifndef INCLUDE_TRANSFORMER2_H_
#define INCLUDE_TRANSFORMER2_H_


#include <string>
#include <map>
#include <array>
#include <vector>

#include "TTree.h"

#include "clipper.hpp"

using namespace std;
using namespace ClipperLib;

#define PIXEL_NUM 200
#define delta 1600
#define MAX_RECHITS 20000



typedef struct Pixel
{
    Path* path;
    double energy;
    Pixel():path(nullptr),energy(0.0) { }
} Pixel;

class Transformer2 {
public:
	Transformer2(string geometry_file, TTree* tree);
	virtual ~Transformer2();
	void transform();

private:
	TTree* tree;

	map<unsigned int, Path> hex;
	array<array<Pixel, PIXEL_NUM>, PIXEL_NUM>  pix;

	std::vector<unsigned int> *rechit_detid;
	std::vector<float>   *rechit_energy;
	std::vector<float>   *rechit_x;
	std::vector<float>   *rechit_y;
	std::vector<float>   *rechit_z;

	TBranch        *b_rechit_detid;
	TBranch        *b_rechit_energy;
	TBranch        *b_rechit_x;
	TBranch        *b_rechit_y;
	TBranch        *b_rechit_z;

	float pixel_x_[MAX_RECHITS];
	float pixel_y_[MAX_RECHITS];
	float pixel_e_[MAX_RECHITS];


	Long64_t LoadTree(Long64_t entry);
	map<unsigned int, Path> read_geo(string geometry_file);
	vector<Pixel*> get_pixels_from_hex(Path hex);
	void split_hexagon_energy(Path h, float energy);
	vector<int> get_index_sequence(int from, int to);
	vector<Pixel*> get_pixels_from_index(vector<int> idx, vector<int> idy);
};

#endif /* INCLUDE_TRANSFORMER2_H_ */
