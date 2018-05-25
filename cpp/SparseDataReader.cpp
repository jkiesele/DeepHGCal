//
// Created by srq2 on 5/17/18.
//

#include "SparseDataReader.h"
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include "TString.h"
#include <string>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include <cmath>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>


using namespace std;

template <typename T>
vector<size_t> SparseDataReader::argsort(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

SparseDataReader::SparseDataReader(std::string inputFile, size_t maxEntries) : inputFile(inputFile), maxEntries(maxEntries) {

}

vector<vector<size_t>> SparseDataReader::findNeighborMatrix(vector<float>& rechit_x, vector<float>& rechit_y, const vector<float>& rechit_z) {
    vector<vector<size_t>>neighborMatrix;
    for (size_t i = 0; i  < rechit_x.size(); i++) {
        vector<float> distances(rechit_x.size());
//        for (size_t j = 0; j < rechit_x.size(); j++) {
//            distances[j] = pow(rechit_x[i] - rechit_x[j], 2) + pow(rechit_y[i] - rechit_y[j], 2) + pow(rechit_z[i] - rechit_z[j], 2);
//        }
//        neighborMatrix.push_back(argsort(distances));
    }
    return neighborMatrix;
}


p::tuple SparseDataReader::execute() {
    open();
    size_t numEvents = tree->GetEntries();

    cout<<"Found events: "<<numEvents<<endl;

    vector<vector<float>> _dataAllFeatures;
    vector<vector<float>> _dataSpatialFeatures;
    vector<vector<float>> _dataSpatialLocalFeatures;
    vector<vector<int64_t >> _dataLabelsOneHot;
    vector<int64_t> _dataNumEntries;

    size_t numEventsInserted = 0;
    for(; numEventsInserted < numEvents; numEventsInserted++) {
        tree->LoadTree(numEventsInserted);
        tree->GetEntry(numEventsInserted);

        assert(root_rechit_z->size() == root_rechit_y->size() and
               root_rechit_z->size() == root_rechit_x->size() and
               root_rechit_z->size() == root_rechit_energy->size() and
               root_rechit_z->size() == root_rechit_vxy->size() and
               root_rechit_z->size() == root_rechit_vz->size());

        size_t n = min(root_rechit_x->size(), maxEntries);

        assert(n!=0);

        vector<float> rechit_x(root_rechit_x->begin(), root_rechit_x->begin() + n);
        vector<float> rechit_y(root_rechit_y->begin(), root_rechit_y->begin() + n);
        vector<float> rechit_z(root_rechit_z->begin(), root_rechit_z->begin() + n);
        vector<float> rechit_energy(root_rechit_energy->begin(), root_rechit_energy->begin() + n);
        vector<float> rechit_vxy(root_rechit_vxy->begin(), root_rechit_vxy->begin() + n);
        vector<float> rechit_vz(root_rechit_vz->begin(), root_rechit_vz->begin() + n);

        vector<float> dataAllFeatures(maxEntries * 4);
        vector<float> dataSpatialFeatures(maxEntries * 3);
        vector<float> dataSpatialLocalFeatures(maxEntries * 2);
        vector<int64_t > dataNeighborMatrix(maxEntries * 10);
        vector<int64_t > dataLabelsOneHot(6);

        for(size_t j = 0; j < n ;j++) {
            dataAllFeatures[j*4 + 0] = rechit_x[j];
            dataAllFeatures[j*4 + 1] = rechit_y[j];
            dataAllFeatures[j*4 + 2] = rechit_z[j];
            dataAllFeatures[j*4 + 3] = rechit_energy[j];

            dataSpatialFeatures[j*3 + 0] = rechit_x[j];
            dataSpatialFeatures[j*3 + 1] = rechit_y[j];
            dataSpatialFeatures[j*3 + 2] = rechit_z[j];

            dataSpatialLocalFeatures[j*2 + 0] = rechit_vxy[j];
            dataSpatialLocalFeatures[j*2 + 1] = rechit_vz[j];
        }

        dataLabelsOneHot[0] = (int64_t) (root_isElectron);
        dataLabelsOneHot[1] = (int64_t) (root_isMuon);
        dataLabelsOneHot[2] = (int64_t) (root_isPionCharged);
        dataLabelsOneHot[3] = (int64_t) (root_isPionNeutral);
        dataLabelsOneHot[4] = (int64_t) (root_isK0Long);
        dataLabelsOneHot[5] = (int64_t) (root_isK0Short);

        assert(root_isElectron + root_isMuon + root_isPionCharged + root_isPionNeutral + root_isK0Short + root_isK0Long == 1);

        _dataNumEntries.push_back(n);
        _dataAllFeatures.push_back(dataAllFeatures);
        _dataSpatialFeatures.push_back(dataSpatialFeatures);
        _dataSpatialLocalFeatures.push_back(dataSpatialLocalFeatures);
        _dataLabelsOneHot.push_back(dataLabelsOneHot);

        cout<<"Event "<<numEventsInserted<<endl;
    }

    np::ndarray return_all = np::zeros(p::make_tuple(numEventsInserted,maxEntries, 4), np::dtype::get_builtin<float>());
    np::ndarray return_spatial = np::zeros(p::make_tuple(numEventsInserted,maxEntries, 3), np::dtype::get_builtin<float>());
    np::ndarray return_spatial_local = np::zeros(p::make_tuple(numEventsInserted,maxEntries, 2), np::dtype::get_builtin<float>());
    np::ndarray return_labels_one_hot = np::zeros(p::make_tuple(numEventsInserted, 6), np::dtype::get_builtin<int64_t>());
    np::ndarray return_num_entries = np::zeros(p::make_tuple(numEventsInserted, 1), np::dtype::get_builtin<int64_t>());

    float* __dataAllFeatures = reinterpret_cast<float*>(return_all.get_data());
    float* __dataSpatialFeatures = reinterpret_cast<float *>(return_spatial.get_data());
    float* __dataSpatialLocalFeatures = reinterpret_cast<float *>(return_spatial_local.get_data());
    int64_t * __dataLabelsOneHot = reinterpret_cast<int64_t*>(return_labels_one_hot.get_data());
    int64_t * __dataNumEntries = reinterpret_cast<int64_t*>(return_num_entries.get_data());

    for (size_t i = 0; i < numEventsInserted; i++) {
        float* offsetAll = __dataAllFeatures + maxEntries * 4 * i;
        float* offsetSpatial = __dataSpatialFeatures + maxEntries * 3 * i;
        float* offsetSpatialLocal = __dataSpatialLocalFeatures + maxEntries * 2 * i;
        int64_t* offsetLabelsOneHot = __dataLabelsOneHot + 6 * i;

        std::copy(_dataAllFeatures[i].begin(), _dataAllFeatures[i].end(), offsetAll);
        std::copy(_dataSpatialFeatures[i].begin(), _dataSpatialFeatures[i].end(), offsetSpatial);
        std::copy(_dataSpatialLocalFeatures[i].begin(), _dataSpatialLocalFeatures[i].end(), offsetSpatialLocal);
        std::copy(_dataLabelsOneHot[i].begin(), _dataLabelsOneHot[i].end(), offsetLabelsOneHot);
        __dataNumEntries[i] = _dataNumEntries[i];
    }

    return p::make_tuple(return_all, return_spatial, return_spatial_local, return_labels_one_hot, return_num_entries);
}

void SparseDataReader::open() {
    TFile * f=new TFile(inputFile.c_str(),"READ");
    if(!f || f->IsZombie()){
        std::cerr << "Input file not found" <<std::endl;
        if(f->IsZombie())
            delete f;
        return;
    }

    tree= (TTree*)f->Get("B4");

    if(!tree || tree->IsZombie()){
        std::cerr << "Input tree not found" <<std::endl;
        f->Close();
        delete f;
        return;
    }

    root_rechit_x = root_rechit_y = root_rechit_z = 0;
    root_rechit_layer = 0;
    root_rechit_energy = 0;
    root_rechit_vxy = 0;
    root_rechit_vz = 0;
    root_isElectron = root_isMuon = root_isPionCharged = root_isPionNeutral = root_isK0Long = root_isK0Short = 0;

    tree->SetBranchStatus("*",0);
    tree->SetBranchStatus("isElectron", 1);
    tree->SetBranchStatus("isMuon", 1);
    tree->SetBranchStatus("isPionCharged", 1);
    tree->SetBranchStatus("isPionNeutral", 1);
    tree->SetBranchStatus("isK0Long", 1);
    tree->SetBranchStatus("isK0Short", 1);

    tree->SetBranchStatus("rechit_energy", 1);
    tree->SetBranchStatus("rechit_x", 1);
    tree->SetBranchStatus("rechit_y", 1);
    tree->SetBranchStatus("rechit_z", 1);
    tree->SetBranchStatus("rechit_layer", 1);

    tree->SetBranchStatus("rechit_vxy", 1);
    tree->SetBranchStatus("rechit_vz", 1);

    tree->SetBranchAddress("isElectron", &root_isElectron);
    tree->SetBranchAddress("isMuon", &root_isMuon);
    tree->SetBranchAddress("isPionCharged", &root_isPionCharged);
    tree->SetBranchAddress("isPionNeutral", &root_isPionNeutral);
    tree->SetBranchAddress("isK0Long", &root_isK0Long);
    tree->SetBranchAddress("isK0Short", &root_isK0Short);

    tree->SetBranchAddress("rechit_energy", &root_rechit_energy);
    tree->SetBranchAddress("rechit_x", &root_rechit_x);
    tree->SetBranchAddress("rechit_y", &root_rechit_y);
    tree->SetBranchAddress("rechit_z", &root_rechit_z);
    tree->SetBranchAddress("rechit_vxy", &root_rechit_vxy);
    tree->SetBranchAddress("rechit_vz", &root_rechit_vz);
    tree->SetBranchAddress("rechit_layer", &root_rechit_layer);

}