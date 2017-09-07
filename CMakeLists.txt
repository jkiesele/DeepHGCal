cmake_minimum_required(VERSION 3.8)
project(Converter)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++11 -m64 -I/home/user/Downloads/buildroot/include -L/home/user/Downloads/buildroot/lib -lGui -l Core -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -pthread -lm -ldl -rdynamic")
include_directories(include)
include_directories(src)
include_directories(bin)


set(SOURCE_FILES
        bin/convert.cpp
        include/converter.h
        include/helpers.h
        include/HGCalSel.h
        include/ntuple_content.h
        include/ntuple_globals.h
        include/ntuple_recHits.h
        include/seedMaker.h
        src/converter.cpp
        src/helpers.cpp
        src/HGCalSel.cpp
        src/ntuple_content.cpp
        src/ntuple_globals.cpp
        src/ntuple_recHits.cpp
        src/seedMaker.cpp)

add_executable(Converter ${SOURCE_FILES})