CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS := `root-config --cflags --glibs` 
CC_FLAGS := -fPIC -Wall `root-config --cflags`
CC_FLAGS += -I./include -O2 



all: $(patsubst bin/%.cpp, %, $(wildcard bin/*.cpp)) libhgcalconverter.so



%: bin/%.cpp Makefile $(OBJ_FILES)
	g++ $(CC_FLAGS) $(LD_FLAGS) $(OBJ_FILES) $< -o exe/$@ 

libhgcalconverter.so: $(OBJ_FILES)
	g++ -shared $(LD_FLAGS) -o $@ $^

obj/%.o: src/%.cpp
	g++ $(CC_FLAGS) -c -o $@ $<


clean: 
	rm -f obj/*.o obj/*.d
	rm -f exe/*
	touch bin/*
