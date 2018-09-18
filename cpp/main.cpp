//allows functions with 18 or less paramenters
#define BOOST_PYTHON_MAX_ARITY 20
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numpy.hpp"
#include "boost/python/list.hpp"
#include "boost/python/str.hpp"
//#include "boost/filesystem.hpp"
#include <iostream>
#include <stdint.h>
#include <boost/python/numpy.hpp>
#include "SparseDataReader.h"
#include "GenericDataReader.h"
#include <RechitsMerger.h>


namespace p = boost::python;
namespace np = boost::python::numpy;


p::tuple readSparseData(p::str filename,
                    int max_entries) {

    char* sfilenamet = p::extract<char*>(filename);
    std::string sfilename(sfilenamet);

    return SparseDataReader(sfilename, max_entries).execute();

}

p::tuple readGenericData(p::str file_path, p::str location, p::list branches, p::list types, p::list max_size) {
    return GenericDataReader(file_path, location, branches, types, max_size).execute();
}


p::tuple mergeTwoArrays(np::ndarray arrayA, np::ndarray arrayB, np::ndarray idsA, np::ndarray idsB, np::ndarray sizesA, np::ndarray sizesB) {
    return RechitsMerger(arrayA, arrayB, idsA, idsB, sizesA, sizesB).executeMergeInOneBranch();

}


p::tuple mergeTwoArraysSeparate(np::ndarray arrayA, np::ndarray arrayB, np::ndarray idsA, np::ndarray idsB, np::ndarray sizesA, np::ndarray sizesB) {
    return RechitsMerger(arrayA, arrayB, idsA, idsB, sizesA, sizesB).executeMergeInSeparateBranches();

}

BOOST_PYTHON_MODULE(sparse_hgcal) {
    Py_Initialize();
    np::initialize();
    p::def("read_sparse_data", readSparseData);
    p::def("read_np_array", readGenericData);
    p::def("merge_two_arrays", mergeTwoArrays);
    p::def("merge_two_arrays_separate", mergeTwoArraysSeparate);
}