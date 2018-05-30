//
// Created by srq2 on 5/30/18.
//

#include <iostream>
#include "GenericDataReader.h"
#include "Helpers.h"
#include <string>
#include <TTreeReader.h>
#include <memory>
#include <algorithm>
#include <math.h>
#include <TFile.h>

using namespace std;

GenericDataReader::GenericDataReader(const boost::python::str &file_name, const boost::python::str &location,
                                     const boost::python::list &branches, const boost::python::list &types,
                                     const boost::python::list &max_size) : file_name(file_name), location(location),
                                                                            branches(branches), types(types),
                                                                            max_size(max_size) {}


void GenericDataReader::makeResultArrays(const size_t& maxEvents) {

    assert(p::len(types) == p::len(branches));
    assert(p::len(types) == p::len(max_size));

    vector<np::ndarray>& output = arrays;
    vector<np::ndarray>& outputSizes = arraySizes;

    for (int i = 0; i < p::len(types); i++) {
        string type = str2string(p::extract<p::str>(types[i]));
        int maxSize = p::extract<int>(max_size[i]);
        bool isArray = maxSize > 1;

        if (type == "float32") {
            if (isArray) {
                output.push_back(np::zeros(p::make_tuple(maxEvents, maxSize), np::dtype::get_builtin<float>()));
            }
            else {
                output.push_back(np::zeros(p::make_tuple(maxEvents), np::dtype::get_builtin<float>()));
            }
        }
        else if (type == "float64") {
            if (isArray) {
                output.push_back(np::zeros(p::make_tuple(maxEvents, maxSize), np::dtype::get_builtin<double>()));
            }
            else {
                output.push_back(np::zeros(p::make_tuple(maxEvents), np::dtype::get_builtin<double>()));
            }
        }
        else if (type == "int32") {
            if (isArray) {
                output.push_back(np::zeros(p::make_tuple(maxEvents, maxSize), np::dtype::get_builtin<int32_t>()));
            }
            else {
                output.push_back(np::zeros(p::make_tuple(maxEvents), np::dtype::get_builtin<int32_t>()));
            }

        }
        else if (type == "int64") {
            if (isArray) {
                output.push_back(np::zeros(p::make_tuple(maxEvents, maxSize), np::dtype::get_builtin<int64_t>()));
            }
            else {
                output.push_back(np::zeros(p::make_tuple(maxEvents), np::dtype::get_builtin<int64_t>()));
            }
        }
        else {
            cout<<"Error: Unknown type."<<endl;
            throw -1; // TODO: Change to exception later?

        }
        outputSizes.push_back(np::zeros(p::make_tuple(maxEvents), np::dtype::get_builtin<int64_t>()));
    }
}
void GenericDataReader::makeValueReaders(TTreeReader& reader, vector<void *> &valueReaders) {

    assert(p::len(types) == p::len(branches));
    assert(p::len(types) == p::len(max_size));

    for (int i = 0; i < p::len(types); i++) {
        string branchName = str2string(p::extract<p::str>(branches[i]));
        string type = str2string(p::extract<p::str>(types[i]));
        int maxSize = p::extract<int>(max_size[i]);
        bool isArray = maxSize > 1;

        if (type == "float32") {
            if (isArray) {
                TTreeReaderValue<std::vector<float>>* valueReader = new TTreeReaderValue<std::vector<float >>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
            else {
                TTreeReaderValue<float>* valueReader = new TTreeReaderValue<float>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
        }
        else if (type == "float64") {
            if (isArray) {
                TTreeReaderValue<std::vector<double>>* valueReader = new TTreeReaderValue<std::vector<double >>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
            else {
                TTreeReaderValue<double>* valueReader = new TTreeReaderValue<double>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
        }
        else if (type == "int32") {
            if (isArray) {
                TTreeReaderValue<std::vector<int32_t>>* valueReader = new TTreeReaderValue<std::vector<int32_t >>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
            else {
                TTreeReaderValue<int32_t>* valueReader = new TTreeReaderValue<int32_t>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }

        }
        else if (type == "int64") {
            if (isArray) {
                TTreeReaderValue<std::vector<int64_t>>* valueReader = new TTreeReaderValue<std::vector<int64_t >>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
            else {
                TTreeReaderValue<int64_t>* valueReader = new TTreeReaderValue<int64_t>(reader, branchName.c_str());
                valueReaders.push_back(valueReader);
            }
        }
    }
}

void GenericDataReader::freeValueReaders(TTreeReader& reader, vector<void *> &readers) {

    assert(p::len(types) == p::len(branches));
    assert(p::len(types) == p::len(max_size));

    for (int i = 0; i < p::len(types); i++) {
        string branchName = str2string(p::extract<p::str>(branches[i]));
        string type = str2string(p::extract<p::str>(types[i]));
        int maxSize = p::extract<int>(max_size[i]);
        bool isArray = maxSize > 1;

        if (type == "float32") {
            if (isArray) {
                TTreeReaderValue<std::vector<float>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<float>>*>(readers[i]);
                delete valueReader;
            }
            else {
                TTreeReaderValue<float>* valueReader = reinterpret_cast<TTreeReaderValue<float>*>(readers[i]);
                delete valueReader;
            }
        }
        else if (type == "float64") {
            if (isArray) {
                TTreeReaderValue<std::vector<double>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<double>>*>(readers[i]);
                delete valueReader;
            }
            else {
                TTreeReaderValue<double>* valueReader = reinterpret_cast<TTreeReaderValue<double>*>(readers[i]);
                delete valueReader;
            }
        }
        else if (type == "int32") {
            if (isArray) {
                TTreeReaderValue<std::vector<int32_t>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<int32_t>>*>(readers[i]);
                delete valueReader;
            }
            else {
                TTreeReaderValue<int32_t>* valueReader = reinterpret_cast<TTreeReaderValue<int32_t>*>(readers[i]);
                delete valueReader;
            }

        }
        else if (type == "int64") {
            if (isArray) {
                TTreeReaderValue<std::vector<int64_t>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<int64_t>>*>(readers[i]);
                delete valueReader;
            }
            else {
                TTreeReaderValue<int64_t>* valueReader = reinterpret_cast<TTreeReaderValue<int64_t>*>(readers[i]);
                delete valueReader;
            }
        }
    }
}

void GenericDataReader::fillFromValueReaders(TTreeReader& reader, vector<void *> &readers, size_t eventNumber) {

    assert(p::len(types) == p::len(branches));
    assert(p::len(types) == p::len(max_size));

    for (int i = 0; i < p::len(types); i++) {
        string branchName = str2string(p::extract<p::str>(branches[i]));
        string type = str2string(p::extract<p::str>(types[i]));
        int maxSize = p::extract<int>(max_size[i]);
        bool isArray = maxSize != 1;
        int64_t* __dataSizesStart = reinterpret_cast<int64_t*>(arraySizes[i].get_data());

        if (type == "float32") {
            float* __dataStart = reinterpret_cast<float*>(arrays[i].get_data());
            float* offset = __dataStart + eventNumber * maxSize;
            if (isArray) {
                TTreeReaderValue<std::vector<float>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<float>>*>(readers[i]);
                std::vector<float>* dataRoot = valueReader->Get();
                int n = min((int)dataRoot->size(), maxSize);
                std::copy(dataRoot->begin(), dataRoot->begin() + n, offset);
                __dataSizesStart[eventNumber] = n;

            }
            else {
                TTreeReaderValue<float>* valueReader = reinterpret_cast<TTreeReaderValue<float>*>(readers[i]);
                float dataRoot = *valueReader->Get();
                *offset = dataRoot;
                __dataSizesStart[eventNumber] = 1;
            }
        }
        else if (type == "float64") {
            double* __dataStart = reinterpret_cast<double*>(arrays[i].get_data());
            double* offset = __dataStart + eventNumber * maxSize;
            if (isArray) {
                TTreeReaderValue<std::vector<double>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<double>>*>(readers[i]);
                std::vector<double>* dataRoot = valueReader->Get();
                int n = min((int)dataRoot->size(), maxSize);
                std::copy(dataRoot->begin(), dataRoot->begin() + n, offset);
                __dataSizesStart[eventNumber] = n;
            }
            else {
                TTreeReaderValue<double>* valueReader = reinterpret_cast<TTreeReaderValue<double>*>(readers[i]);
                double dataRoot = *valueReader->Get();
                *offset = dataRoot;
                __dataSizesStart[eventNumber] = 1;
            }
        }
        else if (type == "int32") {
            int32_t* __dataStart = reinterpret_cast<int32_t*>(arrays[i].get_data());
            int32_t* offset = __dataStart + eventNumber * maxSize;
            if (isArray) {
                TTreeReaderValue<std::vector<int32_t>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<int32_t>>*>(readers[i]);
                std::vector<int32_t>* dataRoot = valueReader->Get();
                int n = min((int)dataRoot->size(), maxSize);
                std::copy(dataRoot->begin(), dataRoot->begin() + n, offset);
                __dataSizesStart[eventNumber] = n;
            }
            else {
                TTreeReaderValue<int32_t>* valueReader = reinterpret_cast<TTreeReaderValue<int32_t>*>(readers[i]);
                int32_t dataRoot = *valueReader->Get();
                *offset = dataRoot;
                __dataSizesStart[eventNumber] = 1;
            }

        }
        else if (type == "int64") {
            int64_t* __dataStart = reinterpret_cast<int64_t*>(arrays[i].get_data());
            int64_t* offset = __dataStart + eventNumber * maxSize;
            if (isArray) {
                TTreeReaderValue<std::vector<int64_t>>* valueReader = reinterpret_cast<TTreeReaderValue<std::vector<int64_t>>*>(readers[i]);
                std::vector<int64_t>* dataRoot = valueReader->Get();
                int n = min((int)dataRoot->size(), maxSize);
                std::copy(dataRoot->begin(), dataRoot->begin() + n, offset);
                __dataSizesStart[eventNumber] = n;
            }
            else {
                TTreeReaderValue<int64_t>* valueReader = reinterpret_cast<TTreeReaderValue<int64_t>*>(readers[i]);
                int64_t dataRoot = *valueReader->Get();
                *offset = dataRoot;
                __dataSizesStart[eventNumber] = 1;
            }
        }
    }
}


p::tuple GenericDataReader::execute() {
    string inputFile = str2string(file_name);
    TFile * f=new TFile(inputFile.c_str(),"READ");
    if(!f || f->IsZombie()){
        std::cerr << "Input file not found" <<std::endl;
        if(f->IsZombie())
            delete f;
        return p::tuple();
    }

    TTreeReader reader(str2string(location).c_str(), f);
    size_t numEvents = reader.GetEntries(false);

    makeResultArrays(numEvents);
    vector<void*> valueReaders;
    makeValueReaders(reader, valueReaders);

    int eventNo = 0;
    while (reader.Next()) {
        fillFromValueReaders(reader, valueReaders, eventNo);
        eventNo++;
    }
    boost::python::list returnListData;
    boost::python::list returnListSizes;
    for (auto &i : arrays) {
        returnListData.append(i);
    }
    for (auto &i : arraySizes) {
        returnListSizes.append(i);
    }
    return p::make_tuple(returnListData, returnListSizes);

//    freeValueReaders(reader, valueReaders);
}