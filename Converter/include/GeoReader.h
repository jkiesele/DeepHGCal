//============================================================================
// Name        : GeoREader.h
// Author      : Ofir Arzi
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#ifndef GEOREADER_H_
#define GEOREADER_H_

#include "../include/clipper.hpp"

#include <map>
#include <vector>

using namespace ClipperLib;
using namespace std;


class GeoReader {
public:
	GeoReader(string file_name);

	Paths read(int layer);

	virtual ~GeoReader();

private:
	string geo; // file name of the geometry file


	//read all the geometry from the file, for the given layer
	Path line_to_hex(vector<string> layer);
};


#endif /* GEOREADER_H_ */
