# location of the Python header files

PYTHON_VERSION = 2.7
# location of the Boost Python include files and library
PYTHON_INCLUDE = ${CONDA_PREFIX}/include/python2.7
PYTHON_LIB=${CONDA_PREFIX}/lib/python$(PYTHON_VERSION)/config
# also works on gpu in compiled version
# this is just luck, ...
BOOST_INC = ${CONDA_PREFIX}/include
BOOST_LIB = ${CONDA_PREFIX}/lib
DEEPJETINCL=${DEEPJETCORE}/compiled/interface
DEEPJETLIB=${DEEPJETCORE}/compiled/

ifdef CMSSW_BASE
	PYTHON_BASE=$(shell scram tool info python | grep BASE | cut -d "=" -f 2)
	PYTHON_INCLUDE=${PYTHON_BASE}/include/python${PYTHON_VERSION}/
	PYTHON_LIB=${PYTHON_BASE}/lib/python${PYTHON_VERSION}/config
	BOOST_BASE=$(shell scram tool info boost | grep BASE | cut -d "=" -f 2)
	BOOST_INC = ${BOOST_BASE}/include
	BOOST_LIB = ${BOOST_BASE}/lib
endif

LINUXADD=-Wl,--export-dynamic
ROOTSTUFF=`root-config --cflags --libs --glibs` -g
CFLAGS=

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

MODULES := $(wildcard src/*.C)
MODULES_OBJ_FILES := $(addprefix ./,$(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./,$(notdir $(MODULES:.C=.so)))

UNAME_S := $(shell uname -s)
# remove linux flags in osx
ifeq ($(UNAME_S),Darwin)
	LINUXADD=""
endif


all: $(MODULES_SHARED_LIBS)

#helpers

obj/%.o: src/%.cpp
	g++ $(CFLAGS) $(ROOTSTUFF) -I./interface -I$(DEEPJETINCL) -O2 -fPIC -c -o $@ $< 
    
#pack helpers in lib
libdeephgcalhelpers.so: $(OBJ_FILES)
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) obj/*.o -o $@ 
  

%.so: %.o libdeephgcalhelpers.so
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) -L$(DEEPJETLIB) -ldeepjetcorehelpers -L$(BOOST_LIB)  -lboost_python -L${PYTHON_LIB} -lpython2.7  $< -o $(@) 


%.o: src/%.C 
	g++   $(ROOTSTUFF) -O2 -I./interface -I$(DEEPJETINCL) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -o $(@) $<
	
clean:
	rm -f libdeephgcalhelpers.so c_createRecHitMap.so