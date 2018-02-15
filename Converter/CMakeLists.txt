cmake_minimum_required(VERSION 3.5)
project(Converter)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
MESSAGE(STATUS "ROOTSYS $ENV{ROOTSYS}")
find_package(ROOT REQUIRED COMPONENTS RIO Net)
include(${ROOT_USE_FILE})

include_directories(include)


set(SOURCE_FILES include/clipper.hpp
        include/converter.h
        include/GeoReader.h
        include/helpers.h
        include/HGCalSel.h
        include/mergeDescriptor.h
        include/ntuple_config.h
        include/ntuple_content.h
        include/ntuple_globals.h
        include/ntuple_recHits.h
        include/seedMaker.h
        include/Transformer.h
        include/Transformer2.h
        include/truthCreator.h
        src/clipper.cpp
        src/converter.cpp
        src/GeoReader.cpp
        src/helpers.cpp
        src/HGCalSel.cpp
        src/mergeDescriptor.cpp
        src/ntuple_content.cpp
        src/ntuple_globals.cpp
        src/ntuple_recHits.cpp
        src/seedMaker.cpp
        src/Transformer.cpp
        src/Transformer2.cpp
        src/truthCreator.cpp)



add_executable(convert bin/convert.cpp ${SOURCE_FILES})
add_executable(createMergeList bin/createMergeList.cpp ${SOURCE_FILES})
add_executable(merge bin/merge.cpp ${SOURCE_FILES})


target_link_libraries(convert ${ROOT_LIBRARIES})
target_link_libraries(createMergeList ${ROOT_LIBRARIES})
target_link_libraries(merge ${ROOT_LIBRARIES})
