/*
 * helpers.cpp
 *
 *  Created on: 26 Jun 2017
 *      Author: jkiesele
 */




namespace helpers{
	
	
float deltaPhi(const float& a, const float& b){
	const float pi = 3.14159265358979323846;
	float delta = (a -b);
	while (delta >= pi)  delta-= 2* pi;
	while (delta < -pi)  delta+= 2* pi;
	return delta;
}


	

}