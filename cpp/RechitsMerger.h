//
// Created by srq on 14.09.18.
//

#include <vector>
#include <string>
#include <boost/python/numpy.hpp>
#include "TTree.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

#ifndef DEEPHGCAL2_RECHITSMERGER_H
#define DEEPHGCAL2_RECHITSMERGER_H

class RechitsMerger {
protected:

    float *dataA;
    float *dataB;
    int *idsA;
    int *idsB;
    int *sizesA;
    int *sizesB;

    int batchSize;
    int maxVertices;
    int numFeatures;

public:
    RechitsMerger(np::ndarray arrayA, np::ndarray arrayB, np::ndarray idsA, np::ndarray idsB, np::ndarray sizesA,
                  np::ndarray sizesB);

    p::tuple executeMergeInOneBranch();
    p::tuple executeMergeInSeparateBranches();
};

#endif //DEEPHGCAL2_RECHITSMERGER_H
