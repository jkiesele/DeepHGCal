cmake_minimum_required(VERSION 3.4)
project(DeepHGCalModules)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
MESSAGE(STATUS "ROOTSYS $ENV{ROOTSYS}")
find_package(ROOT REQUIRED COMPONENTS RIO Net)
find_package(Boost 1.40 COMPONENTS regex program_options python REQUIRED )
find_package(PythonLibs 2.7 REQUIRED )

include(${ROOT_USE_FILE})
link_directories(/usr/local/lib/ /usr/lib/ ${Boost_LIBRARY_DIR} ${PYTHON_LIBRARIES} $ENV{DEEPJETCORE}/compiled/build)
include_directories(/usr/local/include/ csv json/json src ${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} $ENV{DEEPJETCORE}/compiled/interface)

set(SOURCE_FILES )

add_library(c_createRecHitMap SHARED src/c_createRecHitMap.C ${SOURCE_FILES})
target_link_libraries(c_createRecHitMap deepjetcorehelpers ${ROOT_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(c_createRecHitMap PROPERTIES PREFIX "")
