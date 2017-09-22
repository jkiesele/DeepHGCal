/*
 * Transformer2.cpp
 *
 *  Created on: 12 Sep 2017
 *      Author: user
 */

#include "../include/Transformer2.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

Transformer2::Transformer2(string geometry_file, TTree* tree) {
	// TODO Auto-generated constructor stub

	hex = read_geo(geometry_file); // TODO: implement

	rechit_detid = 0;
	rechit_energy = 0;
	rechit_x = 0;
	rechit_y = 0;
	rechit_z = 0;

	tree->SetBranchAddress("rechit_detid", &rechit_detid, &b_rechit_detid);
	tree->SetBranchAddress("rechit_energy", &rechit_energy, &b_rechit_energy);
	tree->SetBranchAddress("rechit_x", &rechit_x, &b_rechit_x);
	tree->SetBranchAddress("rechit_y", &rechit_y, &b_rechit_y);
	tree->SetBranchAddress("rechit_z", &rechit_z, &b_rechit_z);

	tree->Branch("pixel_x"  ,&pixel_x_  ,"pixel_x_[n_rechits_]/f" );
	tree->Branch("pixel_y"  ,&pixel_y_  ,"pixel_y_[n_rechits_]/f" );
	tree->Branch("pixel_e"  ,&pixel_e_  ,"pixel_e_[n_rechits_]/f" );

}

Transformer2::~Transformer2() {
	// TODO Auto-generated destructor stub
}


void Transformer2::transform()
{
    tree->SetBranchStatus("rechit_detid",1);
    tree->SetBranchStatus("rechit_energy",1);
	tree->SetBranchStatus("rechit_x",1);
    tree->SetBranchStatus("rechit_y",1);
    tree->SetBranchStatus("rechit_z",1);


    //	1. Iterate over the seeds
    Long64_t nseeds = tree->GetEntries();
    for (Long64_t seed = 0; seed < nseeds; seed++)
    {
        Long64_t ientry = LoadTree(seed);
        if (ientry < 0)
        	break;

        //	2. For each seed, iterate over the hits
        for(size_t hit = 0;hit < rechit_x->size();hit++)
        {
                //	3. For each hit, use it's detectorId property to find the corresponding hexagon
        		Path h = hex[rechit_detid->at(hit)];

        		//  4. split hexagon's energy among overlapping pixels
        		split_hexagon_energy(h, rechit_energy->at(hit));

		}// end for hit

        //	7. Set the pix_x, pix_y, pix_e branches of the input tree with each sqaure's x,y coordinates and it's energy.
        int count = 0;
        for(auto i: pix)
        {
        	for(auto p: i)
			{
        		if (p.path != nullptr)
        		{
					pixel_x_[count] = p.path->at(0).X;
					pixel_y_[count] = p.path->at(0).Y;
					pixel_e_[count] = p.energy;
					count++;
        		}
			}
        }
        tree->Fill();

        // reinitialize pix for the next seed
        pix = array<array<Pixel, PIXEL_NUM>, PIXEL_NUM>();
	}// end for seed


}//end transform

vector<Pixel*> Transformer2::get_pixels_from_hex(Path h)
{
	vector<int> x;
	vector<int> y;

	for (auto p:h)
	{
		x.push_back(p.X / delta);
		y.push_back(p.Y / delta);
	}

	vector<int> idx = get_index_sequence(*max_element(x.begin(), x.end()), *min_element(x.begin(), x.end()));
	vector<int> idy = get_index_sequence(*max_element(y.begin(), y.end()), *min_element(y.begin(), y.end()));

	vector<Pixel*> pixels = get_pixels_from_index(idx, idy);

	return pixels;
}

vector<int> Transformer2::get_index_sequence(int from, int to)
{
	auto add = [](int &i) { i += 100;};


	vector<int> idx(to - from + 2);
	iota(idx.begin(), idx.end(), from - 1);
	for_each(idx.begin(), idx.end(), add);

	return idx;
}

vector<Pixel*> Transformer2::get_pixels_from_index(vector<int> idx, vector<int> idy)
{
	vector<Pixel*> pixels;

	for(auto col : idx)
	{
		for(auto row : idy)
		{
			if(pix[row][col].path == nullptr)
			{
				Path *temp = new Path();
				*temp << IntPoint((col - 100) * delta, (row - 100) * delta) << IntPoint((col - 100) * delta, (row - 99) * delta) <<
						 IntPoint((col - 99) * delta, (row - 99) * delta) << IntPoint((col - 99) * delta, (row - 100) * delta);
				pix[row][col].path = temp;

			}
			pixels.push_back(&pix[row][col]);
		}
	}

	return pixels;
}
void Transformer2::split_hexagon_energy(Path h, float energy)
{
	//	5. Find the overlapping square pixels. - TODO implement
	vector<Pixel*> pixels = get_pixels_from_hex(h);

    //	6. Use a clipper to find the energy contribution of the hexgon to each square.
	Clipper c;
	Paths intersections;


	for (unsigned int i = 0; i < pixels.size();  i++)
	{
		c.AddPath(*(*pixels[i]).path, ptSubject, true);
		c.AddPath(h, ptClip, true);
		c.Execute(ctIntersection, intersections, pftNonZero, pftNonZero);

		for (auto intersection:intersections)
		{
			(*pixels[i]).energy +=  abs( Area(intersection) / Area(h)) * energy;

		}

		c.Clear();
		intersections.clear();

	}// end for pixel

}

Long64_t Transformer2::LoadTree(Long64_t seed)
{
// Set the environment to read one entry

   Long64_t centry = tree->LoadTree(seed);
   if (centry < 0) return centry;

   return centry;
}

map<unsigned int, Path> Transformer2::read_geo(string geometry_file)
{
	//	2. For each layer, create a vector (2d array? map?) of hexagons, ordered by y, then x.
	map<unsigned int, Path> res;

	return res;
}

