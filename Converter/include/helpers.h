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


    /**
     * Converts from cartesian x, y, z coordinates to r in spherical coordiantes
     *
     *
     * @param x x coordinate in cartesian
     * @param y y coordinate in cartesian
     * @param z z coordinate in cartesian
     * @return r in spherical coordinates
     */
    float cartesianToSphericalR(const float &x, const float &y, const float &z);

    /**
     * Compute if the rec hit lies within range of the particle from event
     *
     * @param particleEta eta of the particle
     * @param particlePhi phi of the particle
     * @param eta eta of the rec hit
     * @param phi phi of the rec hit
     * @param dR distance
     * @return
     */
    bool recHitMatchesParticle(const float &particleEta, const float &particlePhi, const float &eta, const float &phi,
                               const float &dR);


}


#endif /* CONVERTER_INCLUDE_HELPERS_H_ */
