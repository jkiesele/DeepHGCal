//
// Created by Shah Rukh Qasim on 5/17/18.
//

#include <vector>
#include <string>
#include <boost/python/numpy.hpp>
#include "TTree.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef CPP_SPARSEDATAREADER_H
#define CPP_SPARSEDATAREADER_H


class SparseDataReader {

public:
    p::tuple execute();
    void open();
    SparseDataReader(std::string inputFile, size_t maxEntries);


private:
    std::string inputFile;
    size_t maxEntries;
    TTree* tree;

    const int maxNeighbors=10;
    float*resultDataAllFeatures;
    float*resultDataSpatialFeatures;
    int*resultDataNeighborMatrix;
    int*resultDataLabelsOneHot;

    int root_isElectron;
    int root_isMuon;
    int root_isPionCharged;
    int root_isPionNeutral;
    int root_isK0Long;
    int root_isK0Short;
    std::vector<float>* root_rechit_energy;
    std::vector<float>* root_rechit_x;
    std::vector<float>* root_rechit_y;
    std::vector<float>* root_rechit_z;
    TBranch *b_temp;

    template <typename T>
    std::vector<size_t> argsort(const std::vector<T> &v);
    std::vector<std::vector<size_t>> findNeighborMatrix(std::vector<float>& rechit_x, std::vector<float>& rechit_y, const std::vector<float>& rechit_z);


};


#endif //CPP_SPARSEDATAREADER_H
