cmake_minimum_required(VERSION 2.8)
project(DeepHGCal2)


set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
MESSAGE(STATUS "ROOTSYS $ENV{ROOTSYS}")

find_package(ROOT REQUIRED RIO Tree MathCore Core Hist Gui Gpad TreeViewer MLP Matrix Thread Graf)

include(${ROOT_USE_FILE})

include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Boost 1.63.0 COMPONENTS system python numpy)
include_directories(${Boost_INCLUDE_DIRS})



find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

set(SOURCE_FILES main.cpp SparseDataReader.cpp SparseDataReader.h)

add_library(sparse_hgcal SHARED ${SOURCE_FILES})

message(Boost libs: ${Boost_LIBRARIES})
message(Root libs: ${ROOT_LIBRARIES})
message(Python libs: ${PYTHON_LIBRARY})

set_target_properties(sparse_hgcal PROPERTIES PREFIX "" )
target_link_libraries(sparse_hgcal ${ROOT_LIBRARIES} ${Boost_LIBRARIES})
