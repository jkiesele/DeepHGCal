/*
 * helpers.h
 *
 *  Created on: 26 Jun 2017
 *      Author: jkiesele
 */

#ifndef CONVERTER_INCLUDE_HELPERS_H_
#define CONVERTER_INCLUDE_HELPERS_H_


namespace helpers {
    float deltaPhi(const float &phi1, const float &phi2);


    /***
     * Compute L2 difference between seed and cluster parameters.
     *
     * @param seedEta Eta of the seed in rapid in pseudorapidity scale
     * @param seedPhi Phi angle of the seed in radians
     * @param clusterEta Eta of the cluster in rapid in pseudorapidity scale
     * @param clusterPhiEta of the cluster in rapid in pseudorapidity scale
     * @return The L2 difference
     */
    float getSeedSimClusterDifference(const float &seedEta, const float &seedPhi, const float &clusterEta,
                                      const float &clusterPhi);
}


#endif /* CONVERTER_INCLUDE_HELPERS_H_ */
