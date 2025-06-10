# === Variables de proyecto ===
TARGET         := attention
EXAMPLE        := example_feedforward
BUILD_DIR      := build
SRC_DIR        := src
EXAMPLES_DIR   := examples

# === Archivos fuente ===
CUDA_SRC       := $(wildcard $(SRC_DIR)/*.cu)
CPP_SRC        := $(wildcard $(SRC_DIR)/**/*.cpp) $(wildcard $(SRC_DIR)/*.cpp)
EXAMPLE_SRC    := $(EXAMPLES_DIR)/example_feedforward.cpp

# === Compiladores y flags ===
NVCC           := nvcc
CXX            := g++
NVCC_FLAGS     := -arch=sm_75 -std=c++17
CXX_FLAGS      := -std=c++17 -O2

# === Targets ===
.PHONY: all clean run_example run_attention tree

all: $(BUILD_DIR)/$(TARGET) $(BUILD_DIR)/$(EXAMPLE)

# Ejecutables
$(BUILD_DIR)/$(TARGET): $(CUDA_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(BUILD_DIR)/$(EXAMPLE): $(EXAMPLE_SRC) $(SRC_DIR)/model/feedforward.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $^ -o $@

# Crear carpeta build si no existe
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Limpiar todo lo generado
clean:
	rm -rf $(BUILD_DIR)

# Ejecutar ejemplo de feedforward
run_example: $(BUILD_DIR)/$(EXAMPLE)
	./$(BUILD_DIR)/$(EXAMPLE)

# Ejecutar attention
run_attention: $(BUILD_DIR)/$(TARGET)
	./$(BUILD_DIR)/$(TARGET)

# Ver estructura de archivos (requiere 'tree')
tree:
	tree -I '$(BUILD_DIR)'

# Ayuda
help:
	@echo "Comandos útiles:"
	@echo "  make                # Compila todo"
	@echo "  make clean          # Limpia la carpeta build"
	@echo "  make run_example    # Ejecuta el ejemplo feedforward"
	@echo "  make run_attention  # Ejecuta el binario attention"
	@echo "  make tree           # Muestra el árbol de archivos"