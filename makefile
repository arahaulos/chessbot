CXX = g++

CXXFLAGS = -Wall -Wextra -std=c++17 -O3 -s -m64 -march=native -funroll-loops
LDFLAGS = -static -static-libgcc -static-libstdc++

ifeq ($(OS),Windows_NT)
    OBJ_FORMAT = pe-x86-64
else
    OBJ_FORMAT = elf64-x86-64
endif


SRCS = $(wildcard *.cpp) $(wildcard chessbot/*.cpp) $(wildcard chessbot/nnue/*.cpp) $(wildcard chessbot/nnue/training/*.cpp) $(wildcard chessbot/util/*.cpp)
OBJS = $(SRCS:.cpp=.o)

BINARY_FILE = embedded_weights.nnue
BINARY_OBJ = embedded_weights.o

TARGET = chessbot_x64

all: $(BINARY_OBJ) $(TARGET)

$(BINARY_OBJ): $(BINARY_FILE)
	objcopy -I binary -O $(OBJ_FORMAT) -B i386:x86-64 $< $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(BINARY_OBJ)
	
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

	