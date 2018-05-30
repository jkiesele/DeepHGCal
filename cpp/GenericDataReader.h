//
// Created by Shah Rukh Qasim on 5/30/18.
//

#include <vector>
#include <string>
#include <boost/python/numpy.hpp>
#include <TTreeReader.h>
#include "TTree.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef DEEPHGCAL2_GENERICDATAREADER_H
#define DEEPHGCAL2_GENERICDATAREADER_H


class GenericDataReader {
private:
    p::str file_name;
    p::str location;
    p::list branches;
    p::list types;
    p::list max_size;
    std::vector<np::ndarray> arrays;
    std::vector<np::ndarray> arraySizes;
public:
    GenericDataReader(const boost::python::str &file_name, const boost::python::str &location,
                      const boost::python::list &branches, const boost::python::list &types,
                      const boost::python::list &max_size);

    p::tuple execute();

private:
    void makeResultArrays(const size_t& maxEvents);
    void makeValueReaders(TTreeReader& reader, std::vector<void*>& valueReaders);
    void fillFromValueReaders(TTreeReader& reader, std::vector<void*>& valueReaders, size_t eventNumber);
    void freeValueReaders(TTreeReader& reader, std::vector<void*>& valueReaders);

};


#endif //DEEPHGCAL2_GENERICDATAREADER_H
