//
// Created by srq2 on 5/30/18.
//

#include "Helpers.h"
#include <string>


std::string str2string(p::str input) {
    char* sfilenamet = p::extract<char*>(input);
    return std::string(sfilenamet);
}