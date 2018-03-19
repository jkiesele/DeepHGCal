CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS := `root-config --cflags --glibs` 
CC_FLAGS := -fPIC -Wall -std=c++11 `root-config --cflags`
CC_FLAGS += -I./include -O2 


execs=$(patsubst %.cpp, exe/%, $(wildcard *.cpp))
all: $(execs) libhgcalconverter.so


exe/%: %.cpp Makefile $(OBJ_FILES)
	g++ $(CC_FLAGS) $(LD_FLAGS) $(OBJ_FILES) $< -o $@ 

libhgcalconverter.so: $(OBJ_FILES)
	g++ -shared $(LD_FLAGS) -o $@ $^

obj/%.o: src/%.cpp
	g++ $(CC_FLAGS) -c -o $@ $<


clean: 
	rm -f obj/*.o obj/*.d
	rm -f $(execs)
	rm -f libhgcalconverter.so