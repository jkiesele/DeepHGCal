/*
 * helpers.cpp
 *
 *  Created on: 26 Jun 2017
 *      Author: jkiesele
 */

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <limits>

using namespace std;

namespace helpers {


    double wrapMax(double x, double max)
    {
        /* integer math: `(max + x % max) % max` */
        return fmod(max + fmod(x, max), max);
    }
    double wrapMinMax(double x, double min, double max)
    {
        return min + wrapMax(x - min, max - min);
    }

    float deltaPhi(const float &a, const float &b) {
        float delta = (a - b);
        delta =  wrapMinMax(delta, -M_PI, +M_PI);
        return delta;
    }

    float deltaPhi2(const float& a, const float& b){
        const float pi = 3.14159265358979323846;
        float delta = (a -b);
        while (delta >= pi)  delta-= 2* pi;
        while (delta < -pi)  delta+= 2* pi;
        return delta;
    }

    float getSeedSimClusterDifference(const float &seedEta, const float &seedPhi, const float &clusterEta,
                                      const float &clusterPhi) {
        float deltaeta = fabs(seedEta-clusterEta);
        float deltaphi = fabs(deltaPhi(seedPhi,clusterPhi));

        return deltaeta*deltaeta + deltaphi*deltaphi;
    }

    float cartesianToSphericalR(const float &x, const float &y, const float &z) {
        return sqrt(x*x + y*y + z*z);
    }


    bool recHitMatchesParticle(const float &particleEta, const float &particlePhi, const float &eta, const float &phi,
                               const float &dR){
        float deltaeta=particleEta-eta;
        if(fabs(deltaeta)>dR)
            return false;
        float deltaphi=helpers::deltaPhi(particlePhi,phi);
        if(fabs(deltaphi)>dR)
            return false;

        float deltaR2=deltaeta*deltaeta + deltaphi*deltaphi;
        if(deltaR2<dR*dR){
            return true;
        }
        return false;

    }


}