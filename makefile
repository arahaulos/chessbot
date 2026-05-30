CXX := clang++

CXXFLAGS_RELEASE := -Wall -std=c++17 -O3 -m64 -march=native -funroll-loops -flto
CXXFLAGS_DEBUG   := -Wall -std=c++17 -march=native -g -Og
LDFLAGS_RELEASE  := -static -static-libgcc -static-libstdc++ -s -flto
LDFLAGS_DEBUG    :=

BINARY_NAME := chessbot_x64


ifeq ($(OS),Windows_NT)
    OBJ_FORMAT = pe-x86-64
else
    OBJ_FORMAT = elf64-x86-64
endif

BINARY_FILE = embedded_weights.nnue
BINARY_OBJ = build/embedded_weights.o

LIBS :=
SRCS := $(wildcard *.cpp) $(wildcard chessbot/*.cpp) $(wildcard chessbot/nnue/*.cpp) $(wildcard chessbot/nnue/training/*.cpp) $(wildcard chessbot/util/*.cpp)
OBJS_RELEASE := $(patsubst %.cpp, build/release/obj/%.o, $(SRCS))
OBJS_DEBUG := $(patsubst %.cpp, build/debug/obj/%.o, $(SRCS))
DEPS := $(OBJS_RELEASE:.o=.d) $(OBJS_DEBUG:.o=.d)

DEPFLAGS := -MMD -MP

TARGET_RELEASE := build/release/bin/$(BINARY_NAME)
TARGET_DEBUG := build/debug/bin/$(BINARY_NAME)

.PHONY: all debug clean

all: $(BINARY_OBJ) $(TARGET_RELEASE)
debug: $(BINARY_OBJ) $(TARGET_DEBUG)

$(BINARY_OBJ): $(BINARY_FILE)
	@mkdir -p $(dir $@)
	objcopy -I binary -O $(OBJ_FORMAT) -B i386:x86-64 $< $@


$(TARGET_RELEASE): $(OBJS_RELEASE)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_RELEASE) $(LDFLAGS_RELEASE) -o $@ $^ $(BINARY_OBJ) $(addprefix -l,$(LIBS))

$(TARGET_DEBUG): $(OBJS_DEBUG)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_DEBUG) $(LDFLAGS_DEBUG) -o $@ $^ $(BINARY_OBJ) $(addprefix -l,$(LIBS))

build/release/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_RELEASE) $(DEPFLAGS) -c $< -o $@

build/debug/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS_DEBUG) $(DEPFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -rf build
