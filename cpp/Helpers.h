//
// Created by Shah Rukh Qasim on 5/30/18.
//


#include <vector>
#include <string>
#include <boost/python/numpy.hpp>
#include "TTree.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef DEEPHGCAL2_HELPERS_H
#define DEEPHGCAL2_HELPERS_H

std::string str2string(p::str input);


#endif //DEEPHGCAL2_HELPERS_H
