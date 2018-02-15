/*
 * helpers.cpp
 *
 *  Created on: 26 Jun 2017
 *      Author: jkiesele
 */

#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

namespace helpers {
    float deltaPhi(const float &a, const float &b) {
        const float pi = 3.14159265358979323846;
        float delta = (a - b);
        while (delta >= pi) delta -= 2 * pi;
        while (delta < -pi) delta += 2 * pi;
        return delta;
    }

    float getSeedSimClusterDifference(const float &seedEta, const float &seedPhi, const float &clusterEta,
                                      const float &clusterPhi) {
        float deltaeta = fabs(seedEta-clusterEta);
        float deltaphi = fabs(deltaPhi(seedPhi,clusterPhi));

        return deltaeta*deltaeta + deltaphi*deltaphi;
    }


}