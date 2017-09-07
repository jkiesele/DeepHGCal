/*
 * GeoReader.cpp
 *
 *  Created on: 7 Sep 2017
 *      Author: user
 */

#include "GeoReader.h"




#include <iostream>

#include <limits>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <fstream>



GeoReader::GeoReader(string file_name): geo(file_name) {}

Paths GeoReader::read(int layer)
{
	//open file
	ifstream geo_file;
	geo_file.open(geo, ios::in );

	//currentl line
	string line;

	//list of the hexagons in layer
	Paths hexes;


	vector<string> tokens;
	while(!geo_file.eof())
	{
		//read line, split to individual values
		getline(geo_file ,line);
		istringstream iss(line);
		tokens = vector<string>{istream_iterator<string>(iss),
		                                  istream_iterator<string>()};
		//layer is hexagon
		if(tokens.size() != 4)
		{
			hexes << line_to_hex(tokens);
		}
		//move on the next layer, until required layer is found
		else if (stoi(tokens[0]) < layer)
		{
			for (int i=0; i <stoi(tokens[1]); i++ )
				getline(geo_file ,line);
		}
		//reading of required layer is over, return
		else if (stoi(tokens[0]) > layer)
		{
			return hexes;
		}

	}

	return hexes;

}

GeoReader::~GeoReader() {}

Path GeoReader::line_to_hex(vector<string> line)
{
	Path hex;
	int x, y;

	//slips first to values (flag and number of vertices)
	// reads the (x,y) values of each vertex.
	for(auto i =line.begin()+2; i != line.end() - 2; i+=2)
	{
		x = stod(*i) * 10;
		y = stod(*(i+1)) * 10;
		hex << IntPoint(x, y);
	}
	cout << endl;

	return hex;
}
