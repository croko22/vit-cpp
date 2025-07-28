CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude

SRC_DIR = src
BUILD_DIR = build
APP_DIR = app

SOURCES = $(wildcard $(SRC_DIR)/core/*.cpp) \
          $(wildcard $(SRC_DIR)/model/*.cpp) \
          $(wildcard $(SRC_DIR)/optimizer/*.cpp)

OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

all: train infer evaluate

train: $(OBJECTS)
	@echo "Enlazando programa de entrenamiento..."
	$(CXX) $(CXXFLAGS) $(APP_DIR)/train.cpp $(OBJECTS) -o $(BUILD_DIR)/train.out

infer: $(OBJECTS)
	@echo "Enlazando programa de inferencia..."
	$(CXX) $(CXXFLAGS) $(APP_DIR)/infer.cpp $(OBJECTS) -o $(BUILD_DIR)/infer.out

evaluate: $(OBJECTS)
	@echo "Enlazando programa de evaluación..."
	$(CXX) $(CXXFLAGS) $(APP_DIR)/evaluate.cpp $(OBJECTS) -o $(BUILD_DIR)/evaluate.out

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compilando $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Limpiando directorio de compilación..."
	rm -rf $(BUILD_DIR)

.PHONY: all train infer evaluate clean
