CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude

SRC_DIR = src
BUILD_DIR = build
APP_DIR = app

CORE_SOURCES = $(wildcard $(SRC_DIR)/core/*.cpp)
MODEL_SOURCES = $(wildcard $(SRC_DIR)/model/*.cpp)
CORE_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SOURCES))
MODEL_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(MODEL_SOURCES))

TRAIN_OBJS = $(BUILD_DIR)/core/activation.o \
			 $(BUILD_DIR)/core/random.o \
			 $(BUILD_DIR)/core/tensor.o \
			 $(BUILD_DIR)/model/vit.o \
			 $(BUILD_DIR)/model/encoder.o \
			 $(BUILD_DIR)/model/layernorm.o \
			 $(BUILD_DIR)/model/linear.o \
			 $(BUILD_DIR)/model/mlp.o

all: train infer

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

train: $(TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) $(APP_DIR)/train.cpp $^ -o $(BUILD_DIR)/train.out

infer: $(TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) $(APP_DIR)/infer.cpp $^ -o $(BUILD_DIR)/infer.out

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all train infer clean