
BOOST_INC = ${CONDA_PREFIX}/include
BOOST_LIB = ${CONDA_PREFIX}/lib

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS := `root-config --cflags --glibs`
LD_FLAGS += -L${BOOST_LIB} -lboost_program_options 
CC_FLAGS := -fPIC -Wall -std=c++11 `root-config --cflags`
CC_FLAGS += -I./include -O2 

$(info $$LD_FLAGS is [${LD_FLAGS}])


execs=$(patsubst %.cpp, exe/%, $(wildcard *.cpp))
all: $(execs) libhgcalconverter.so


exe/%: %.cpp Makefile $(OBJ_FILES)
	g++ $(CC_FLAGS) $(LD_FLAGS) $(OBJ_FILES) -I$(BOOST_INC) $< -o $@ 

libhgcalconverter.so: $(OBJ_FILES)
	g++ -shared $(LD_FLAGS) -o $@ $^

obj/%.o: src/%.cpp
	g++ $(CC_FLAGS) $(LD_FLAGS) -I$(BOOST_INC) -c -o $@ $<


clean: 
	rm -f obj/*.o obj/*.d
	rm -f $(execs)
	rm -f libhgcalconverter.so