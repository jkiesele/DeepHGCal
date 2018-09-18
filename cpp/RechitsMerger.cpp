//
// Created by srq on 14.09.18.
//

#include "RechitsMerger.h"
#include <vector>
#include <unordered_map>
#include <array>
#include <iostream>

using namespace std;


RechitsMerger::RechitsMerger(np::ndarray arrayA, np::ndarray arrayB, np::ndarray idsA, np::ndarray idsB,
                             np::ndarray sizesA, np::ndarray sizesB) {
    /*
     * Common:
     *      B = Batch Size or number of entries in a group of events
     *      V = Max number of vertices
     *      F = Number of features
     *
     * arrayA is the first array, it should be of shape BxVxF
     * arrayB is the second array, it should be of shape BxVxF
     *
     * idsA is a an array of IDs of arrayA. Shape: BxV
     * idsB is an array of IDs of arrayB. Shape: BxV
     *
     * sizesA is an array to represent number of vertices in each batch element of arrayA. Shape: B
     * sizesB is an array to represent number of vertices in each batch element of arrayB. Shape B
     */

    assert(arrayA.get_nd() == 3 and arrayB.get_nd() == 3);
    assert(idsA.get_nd() == 2 and idsB.get_nd() == 2);
    assert(sizesA.get_nd() == 1 and sizesB.get_nd() == 1);

    this->batchSize = arrayA.shape(0);
    this->maxVertices = arrayA.shape(1);
    this->numFeatures = arrayA.shape(2);

    assert(batchSize == arrayB.shape(0));
    assert(maxVertices == arrayB.shape(1));
    assert(numFeatures == arrayB.shape(2));
    assert(batchSize == idsA.shape(0) and batchSize == idsB.shape(0));
    assert(maxVertices == idsA.shape(1) and maxVertices == idsB.shape(1));
    assert(batchSize == sizesA.shape(0) and batchSize == sizesB.shape(0));

    assert(arrayA.get_dtype() == np::dtype::get_builtin<float>());
    assert(arrayB.get_dtype() == np::dtype::get_builtin<float>());
    assert(idsA.get_dtype() == np::dtype::get_builtin<int>());
    assert(idsB.get_dtype() == np::dtype::get_builtin<int>());
    assert(sizesA.get_dtype() == np::dtype::get_builtin<int>());
    assert(sizesB.get_dtype() == np::dtype::get_builtin<int>());


    this->dataA = reinterpret_cast<float*>(arrayA.get_data());
    this->dataB = reinterpret_cast<float*>(arrayB.get_data());
    this->idsA = reinterpret_cast<int*>(idsA.get_data());
    this->idsB = reinterpret_cast<int*>(idsB.get_data());
    this->sizesA = reinterpret_cast<int*>(sizesA.get_data());
    this->sizesB = reinterpret_cast<int*>(sizesB.get_data());
}

p::tuple RechitsMerger::executeMergeInOneBranch() {
    np::ndarray arrayResult = np::zeros(p::make_tuple(this->batchSize, this->maxVertices*2, this->numFeatures), np::dtype::get_builtin<float>());
    np::ndarray _idsResult = np::zeros(p::make_tuple(this->batchSize, this->maxVertices*2), np::dtype::get_builtin<int>());
    np::ndarray _sizesResult = np::zeros(p::make_tuple(this->batchSize), np::dtype::get_builtin<int>());

    float *dataResult = reinterpret_cast<float*>(arrayResult.get_data());
    int *idsResult = reinterpret_cast<int*>(_idsResult.get_data());
    int *sizesResult = reinterpret_cast<int*>(_sizesResult.get_data());

    for (size_t i = 0; i < this->batchSize; i++) {
        size_t iteratorA = 0, iteratorB = 0;
        size_t iteratorOuput = 0;

        while (iteratorA < sizesA[i] and iteratorB < sizesB[i]) {
            size_t idA = idsA[i * this->maxVertices + iteratorA];
            size_t idB = idsB[i * this->maxVertices + iteratorB];

            if (idA < idB) {
                // Add A
                if (iteratorOuput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOuput - 1] == idA) {
                    dataResult[i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures - this->numFeatures] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];
                } else {
                    std::copy(dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures, dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + this->numFeatures,
                              dataResult + i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures);
                    idsResult[i * (this->maxVertices*2) + iteratorOuput] = idA;
                }


                iteratorA++;
                iteratorOuput++;
            } else if (idA > idB) {
                // Add B
                if (iteratorOuput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOuput - 1] == idB) {
                    dataResult[i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures - this->numFeatures] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];
                } else {
                    std::copy(dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures, dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures + this->numFeatures,
                              dataResult + i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures);
                    idsResult[i * (this->maxVertices*2) + iteratorOuput] = idB;
                }
                iteratorB++;
                iteratorOuput++;
            } else {
                // Combine
                std::copy(dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures, dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + this->numFeatures,
                          dataResult + i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures);
                dataResult[i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];
                idsResult[i * (this->maxVertices*2) + iteratorOuput] = idA;

                iteratorA++;
                iteratorB++;
                iteratorOuput++;

            }
        }

        while(iteratorA < sizesA[i]) {
            size_t idA = idsA[i * this->maxVertices + iteratorA];

            // Add A
            if (iteratorOuput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOuput - 1] == idA) {
                dataResult[i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures - this->numFeatures] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];
            } else {
                std::copy(dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures, dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + this->numFeatures,
                          dataResult + i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures);
                idsResult[i * (this->maxVertices*2) + iteratorOuput] = idA;
            }


            iteratorA++;
            iteratorOuput++;
        }

        while(iteratorB < sizesB[i]) {
            size_t idB = idsB[i * this->maxVertices + iteratorB];

            // Add B
            if (iteratorOuput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOuput - 1] == idB) {
                dataResult[i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures - this->numFeatures] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];
            } else {
                std::copy(dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures, dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures + this->numFeatures,
                          dataResult + i * (this->maxVertices*2) * this->numFeatures + iteratorOuput * this->numFeatures);
                idsResult[i * (this->maxVertices*2) + iteratorOuput] = idB;
            }
            iteratorB++;
            iteratorOuput++;

        }

        sizesResult[i] = iteratorOuput;

    }

    return p::make_tuple(arrayResult, _idsResult, _sizesResult);
}


p::tuple RechitsMerger::executeMergeInSeparateBranches() {
    int numFeaturesExtended = numFeatures + 1;

    np::ndarray arrayResult = np::zeros(p::make_tuple(this->batchSize, this->maxVertices*2, numFeaturesExtended), np::dtype::get_builtin<float>());
    np::ndarray _idsResult = np::zeros(p::make_tuple(this->batchSize, this->maxVertices*2), np::dtype::get_builtin<int>());
    np::ndarray _sizesResult = np::zeros(p::make_tuple(this->batchSize), np::dtype::get_builtin<int>());

    float *dataResult = reinterpret_cast<float*>(arrayResult.get_data());
    int *idsResult = reinterpret_cast<int*>(_idsResult.get_data());
    int *sizesResult = reinterpret_cast<int*>(_sizesResult.get_data());

    for (size_t i = 0; i < this->batchSize; i++) {
        size_t iteratorA = 0, iteratorB = 0;
        size_t iteratorOutput = 0;

        while (iteratorA < sizesA[i] and iteratorB < sizesB[i]) {
            size_t idA = idsA[i * this->maxVertices + iteratorA];
            size_t idB = idsB[i * this->maxVertices + iteratorB];

            if (idA < idB) {
                // Add A
                if (iteratorOutput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOutput - 1] == idA) {
                    dataResult[i * (this->maxVertices*2) * numFeaturesExtended + (iteratorOutput-1) * numFeaturesExtended] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];
                } else {
                    std::copy(dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + 1, dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + this->numFeatures,
                              dataResult + i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 2);
                    dataResult[i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];

                    idsResult[i * (this->maxVertices*2) + iteratorOutput] = idA;
                }

                iteratorA++;
                iteratorOutput++;
            } else if (idA > idB) {
                // Add B
                if (iteratorOutput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOutput - 1] == idB) {

                    dataResult[i * (this->maxVertices*2) * numFeaturesExtended + (iteratorOutput-1) * numFeaturesExtended + 1] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];
                } else {
                    std::copy(dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures + 1, dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures + this->numFeatures,
                              dataResult + i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 2);
                    dataResult[i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 1] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];

                    idsResult[i * (this->maxVertices*2) + iteratorOutput] = idB;
                }
                iteratorB++;
                iteratorOutput++;
            } else {
                // Combine
                std::copy(dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + 1, dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + this->numFeatures,
                          dataResult + i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 2);
                dataResult[i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];
                dataResult[i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 1] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];

                idsResult[i * (this->maxVertices*2) + iteratorOutput] = idA;

                iteratorA++;
                iteratorB++;
                iteratorOutput++;

            }
        }

        while(iteratorA < sizesA[i]) {
            size_t idA = idsA[i * this->maxVertices + iteratorA];

            // Add A
            if (iteratorOutput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOutput - 1] == idA) {
                dataResult[i * (this->maxVertices*2) * numFeaturesExtended + (iteratorOutput-1) * numFeaturesExtended] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];
            } else {
                std::copy(dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + 1, dataA + i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures + this->numFeatures,
                          dataResult + i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 2);
                dataResult[i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended] += dataA[i * this->maxVertices * this->numFeatures + iteratorA * this->numFeatures];

                idsResult[i * (this->maxVertices*2) + iteratorOutput] = idA;
            }

            iteratorA++;
            iteratorOutput++;
        }

        while(iteratorB < sizesB[i]) {
            size_t idB = idsB[i * this->maxVertices + iteratorB];

            // Add B
            if (iteratorOutput != 0 and idsResult[i * (this->maxVertices*2) + iteratorOutput - 1] == idB) {
                dataResult[i * (this->maxVertices*2) * numFeaturesExtended + (iteratorOutput-1) * numFeaturesExtended + 1] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];
            } else {
                std::copy(dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures + 1, dataB + i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures + this->numFeatures,
                          dataResult + i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 2);
                dataResult[i * (this->maxVertices*2) * numFeaturesExtended + iteratorOutput * numFeaturesExtended + 1] += dataB[i * this->maxVertices * this->numFeatures + iteratorB * this->numFeatures];

                idsResult[i * (this->maxVertices*2) + iteratorOutput] = idB;
            }
            iteratorB++;
            iteratorOutput++;

        }

        sizesResult[i] = iteratorOutput;

    }

    return p::make_tuple(arrayResult, _idsResult, _sizesResult);
}