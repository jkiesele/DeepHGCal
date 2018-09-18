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
#include <chrono>  // for high_resolution_clock
#include <TTreeReader.h>
#include <TTreeReaderValue.h>


using namespace std;

//template <typename T>
//vector<size_t> SparseDataReader::argsort(const vector<T> &v) {
//
//    // initialize original index locations
//    vector<size_t> idx(v.size());
//    iota(idx.begin(), idx.end(), 0);
//
//    // sort indexes based on comparing values in v
//    sort(idx.begin(), idx.end(),
//         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
//
//    return idx;
//}

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


//p::tuple SparseDataReader::execute() {
//    open();
//    size_t numEvents = tree->GetEntries();
//
//    cout<<"Found events: "<<numEvents<<endl;
//
//    np::ndarray return_all = np::zeros(p::make_tuple(numEvents,maxEntries, 4), np::dtype::get_builtin<float>());
//    np::ndarray return_spatial = np::zeros(p::make_tuple(numEvents,maxEntries, 3), np::dtype::get_builtin<float>());
//    np::ndarray return_spatial_local = np::zeros(p::make_tuple(numEvents,maxEntries, 2), np::dtype::get_builtin<float>());
//    np::ndarray return_labels_one_hot = np::zeros(p::make_tuple(numEvents, 6), np::dtype::get_builtin<int64_t>());
//    np::ndarray return_num_entries = np::zeros(p::make_tuple(numEvents, 1), np::dtype::get_builtin<int64_t>());
//
//    float* __dataAllFeatures = reinterpret_cast<float*>(return_all.get_data());
//    float* __dataSpatialFeatures = reinterpret_cast<float *>(return_spatial.get_data());
//    float* __dataSpatialLocalFeatures = reinterpret_cast<float *>(return_spatial_local.get_data());
//    int64_t * __dataLabelsOneHot = reinterpret_cast<int64_t*>(return_labels_one_hot.get_data());
//    int64_t * __dataNumEntries = reinterpret_cast<int64_t*>(return_num_entries.get_data());
//
//
//    size_t numEventsInserted = 0;
//    for(; numEventsInserted < numEvents; numEventsInserted++) {
//        cout<<"Loading ";
//        auto start = std::chrono::high_resolution_clock::now();
//        tree->GetEntry(numEventsInserted);
//
//        assert(root_rechit_z->size() == root_rechit_y->size() and
//               root_rechit_z->size() == root_rechit_x->size() and
//               root_rechit_z->size() == root_rechit_energy->size() and
//               root_rechit_z->size() == root_rechit_vxy->size() and
//               root_rechit_z->size() == root_rechit_vz->size());
//
//        size_t n = min(root_rechit_x->size(), maxEntries);
//
//        assert(n!=0);
//
//        vector<float> rechit_x(root_rechit_x->begin(), root_rechit_x->begin() + n);
//        vector<float> rechit_y(root_rechit_y->begin(), root_rechit_y->begin() + n);
//        vector<float> rechit_z(root_rechit_z->begin(), root_rechit_z->begin() + n);
//        vector<float> rechit_energy(root_rechit_energy->begin(), root_rechit_energy->begin() + n);
//        vector<float> rechit_vxy(root_rechit_vxy->begin(), root_rechit_vxy->begin() + n);
//        vector<float> rechit_vz(root_rechit_vz->begin(), root_rechit_vz->begin() + n);
//
//        vector<float> dataAllFeatures(maxEntries * 4);
//        vector<float> dataSpatialFeatures(maxEntries * 3);
//        vector<float> dataSpatialLocalFeatures(maxEntries * 2);
//        vector<int64_t > dataNeighborMatrix(maxEntries * 10);
//        vector<int64_t > dataLabelsOneHot(6);
//
//        for(size_t j = 0; j < n ;j++) {
//            dataAllFeatures[j*4 + 0] = rechit_x[j];
//            dataAllFeatures[j*4 + 1] = rechit_y[j];
//            dataAllFeatures[j*4 + 2] = rechit_z[j];
//            dataAllFeatures[j*4 + 3] = rechit_energy[j];
//
//            dataSpatialFeatures[j*3 + 0] = rechit_x[j];
//            dataSpatialFeatures[j*3 + 1] = rechit_y[j];
//            dataSpatialFeatures[j*3 + 2] = rechit_z[j];
//
//            dataSpatialLocalFeatures[j*2 + 0] = rechit_vxy[j];
//            dataSpatialLocalFeatures[j*2 + 1] = rechit_vz[j];
//        }
//
//        dataLabelsOneHot[0] = (int64_t) (root_isElectron);
//        dataLabelsOneHot[1] = (int64_t) (root_isMuon);
//        dataLabelsOneHot[2] = (int64_t) (root_isPionCharged);
//        dataLabelsOneHot[3] = (int64_t) (root_isPionNeutral);
//        dataLabelsOneHot[4] = (int64_t) (root_isK0Long);
//        dataLabelsOneHot[5] = (int64_t) (root_isK0Short);
//
//        assert(root_isElectron + root_isMuon + root_isPionCharged + root_isPionNeutral + root_isK0Short + root_isK0Long == 1);
//
//        float* offsetAll = __dataAllFeatures + maxEntries * 4 * numEventsInserted;
//        float* offsetSpatial = __dataSpatialFeatures + maxEntries * 3 * numEventsInserted;
//        float* offsetSpatialLocal = __dataSpatialLocalFeatures + maxEntries * 2 * numEventsInserted;
//        int64_t* offsetLabelsOneHot = __dataLabelsOneHot + 6 * numEventsInserted;
//
//        std::copy(dataAllFeatures.begin(), dataAllFeatures.end(), offsetAll);
//        std::copy(dataSpatialFeatures.begin(), dataSpatialFeatures.end(), offsetSpatial);
//        std::copy(dataSpatialLocalFeatures.begin(), dataSpatialLocalFeatures.end(), offsetSpatialLocal);
//        std::copy(dataLabelsOneHot.begin(), dataLabelsOneHot.end(), offsetLabelsOneHot);
//        __dataNumEntries[numEventsInserted] = n;
//
//        auto finish = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> elapsed = finish - start;
//        cout << "Event " << numEventsInserted << " " << elapsed.count() << " s\n";;
//    }
//
//    return p::make_tuple(return_all, return_spatial, return_spatial_local, return_labels_one_hot, return_num_entries);
//}

p::tuple SparseDataReader::execute() {
    TFile * f=new TFile(inputFile.c_str(),"READ");
    if(!f || f->IsZombie()){
        std::cerr << "Input file not found" <<std::endl;
        if(f->IsZombie())
            delete f;
        return p::make_tuple();
    }


    TTreeReader reader("B4", f);

    TTreeReaderValue<int> r_isElectron(reader, "isElectron");
    TTreeReaderValue<int> r_isMuon(reader, "isMuon");
    TTreeReaderValue<int> r_isPionCharged(reader, "isPionCharged");
    TTreeReaderValue<int> r_isPionNeutral(reader, "isPionNeutral");
    TTreeReaderValue<int> r_isK0Long(reader, "isK0Long");
    TTreeReaderValue<int> r_isK0Short(reader, "isK0Short");
    TTreeReaderValue<std::vector<double >> r_rechit_energy(reader, "rechit_energy");
    TTreeReaderValue<std::vector<double >> r_rechit_x(reader, "rechit_x");
    TTreeReaderValue<std::vector<double >> r_rechit_y(reader, "rechit_y");
    TTreeReaderValue<std::vector<double >> r_rechit_z(reader, "rechit_z");
    TTreeReaderValue<std::vector<double >> r_rechit_vxy(reader, "rechit_vxy");
    TTreeReaderValue<std::vector<double >> r_rechit_vz(reader, "rechit_vz");

    size_t numEvents = reader.GetEntries(false);

    cout<<"Found events: "<<numEvents<<endl;

    np::ndarray return_all = np::zeros(p::make_tuple(numEvents,maxEntries, 4), np::dtype::get_builtin<float>());
    np::ndarray return_spatial = np::zeros(p::make_tuple(numEvents,maxEntries, 3), np::dtype::get_builtin<float>());
    np::ndarray return_spatial_local = np::zeros(p::make_tuple(numEvents,maxEntries, 2), np::dtype::get_builtin<float>());
    np::ndarray return_labels_one_hot = np::zeros(p::make_tuple(numEvents, 6), np::dtype::get_builtin<int64_t>());
    np::ndarray return_num_entries = np::zeros(p::make_tuple(numEvents, 1), np::dtype::get_builtin<int64_t>());

    float* __dataAllFeatures = reinterpret_cast<float*>(return_all.get_data());
    float* __dataSpatialFeatures = reinterpret_cast<float *>(return_spatial.get_data());
    float* __dataSpatialLocalFeatures = reinterpret_cast<float *>(return_spatial_local.get_data());
    int64_t * __dataLabelsOneHot = reinterpret_cast<int64_t*>(return_labels_one_hot.get_data());
    int64_t * __dataNumEntries = reinterpret_cast<int64_t*>(return_num_entries.get_data());


    size_t numEventsInserted = 0;
    while (reader.Next()) {
        cout<<"Loading ";
        auto start = std::chrono::high_resolution_clock::now();

        vector<double> *root_rechit_x = r_rechit_x.Get();
        vector<double> *root_rechit_y = r_rechit_y.Get();
        vector<double> *root_rechit_z = r_rechit_z.Get();
        vector<double> *root_rechit_energy = r_rechit_energy.Get();
        vector<double> *root_rechit_vxy = r_rechit_vxy.Get();
        vector<double> *root_rechit_vz = r_rechit_vz.Get();
        int root_isK0Short = *r_isK0Short.Get();
        int root_isK0Long = *r_isK0Long.Get();
        int root_isPionNeutral = *r_isPionNeutral.Get();
        int root_isPionCharged = *r_isPionCharged.Get();
        int root_isMuon = *r_isMuon.Get();
        int root_isElectron = *r_isElectron.Get();

        assert(root_rechit_z->size() == root_rechit_y->size() and
               root_rechit_z->size() == root_rechit_x->size() and
               root_rechit_z->size() == root_rechit_energy->size() and
               root_rechit_z->size() == root_rechit_vxy->size() and
               root_rechit_z->size() == root_rechit_vz->size());

        size_t n = min(root_rechit_x->size(), maxEntries);

        assert(n!=0);

        vector<double> rechit_x(root_rechit_x->begin(), root_rechit_x->begin() + n);
        vector<double> rechit_y(root_rechit_y->begin(), root_rechit_y->begin() + n);
        vector<double> rechit_z(root_rechit_z->begin(), root_rechit_z->begin() + n);
        vector<double> rechit_energy(root_rechit_energy->begin(), root_rechit_energy->begin() + n);
        vector<double> rechit_vxy(root_rechit_vxy->begin(), root_rechit_vxy->begin() + n);
        vector<double> rechit_vz(root_rechit_vz->begin(), root_rechit_vz->begin() + n);

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

        float* offsetAll = __dataAllFeatures + maxEntries * 4 * numEventsInserted;
        float* offsetSpatial = __dataSpatialFeatures + maxEntries * 3 * numEventsInserted;
        float* offsetSpatialLocal = __dataSpatialLocalFeatures + maxEntries * 2 * numEventsInserted;
        int64_t* offsetLabelsOneHot = __dataLabelsOneHot + 6 * numEventsInserted;

        std::copy(dataAllFeatures.begin(), dataAllFeatures.end(), offsetAll);
        std::copy(dataSpatialFeatures.begin(), dataSpatialFeatures.end(), offsetSpatial);
        std::copy(dataSpatialLocalFeatures.begin(), dataSpatialLocalFeatures.end(), offsetSpatialLocal);
        std::copy(dataLabelsOneHot.begin(), dataLabelsOneHot.end(), offsetLabelsOneHot);
        __dataNumEntries[numEventsInserted] = n;

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        cout << "Event " << numEventsInserted << " " << elapsed.count() << " s\n";
        numEventsInserted += 1;
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

    cout<<"Defining cache"<<endl;
    tree= (TTree*)f->Get("B4");
    tree->SetCacheSize(3e9);
    tree->SetCacheLearnEntries(10000);

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