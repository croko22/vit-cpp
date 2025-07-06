CXX = g++

# Flags del compilador: C++17, todas las advertencias, y le decimos dónde buscar los headers.
CXXFLAGS = -std=c++17 -Wall -Iinclude

# Directorios
SRC_DIR = src
EXAMPLE_DIR = examples
BUILD_DIR = build

# --- Archivos Fuente y Objeto ---

# Creamos una lista de todos los archivos .cpp de nuestro modelo
# La función wildcard busca todos los archivos que coincidan con el patrón.
MODEL_SOURCES = $(wildcard $(SRC_DIR)/model/*.cpp) $(wildcard $(SRC_DIR)/core/*.cpp)

# Convertimos la lista de fuentes a una lista de archivos objeto (.o) que se guardarán en build/
# Ejemplo: src/core/tensor.cpp -> build/core/tensor.o
MODEL_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(MODEL_SOURCES))

# Definimos los objetos que necesita cada ejecutable. Así evitamos repetirnos.
CORE_OBJS = $(BUILD_DIR)/core/tensor.o $(BUILD_DIR)/core/ops.o $(BUILD_DIR)/core/loss.o $(BUILD_DIR)/core/optimizer.o
LN_OBJS = $(BUILD_DIR)/core/tensor.o $(BUILD_DIR)/model/layernorm.o $(BUILD_DIR)/core/ops.o
MHA_OBJS = $(CORE_OBJS) $(BUILD_DIR)/model/multi_head_attention.o
FF_OBJS = $(CORE_OBJS) $(BUILD_DIR)/model/feedforward.o
ENC_OBJS = $(CORE_OBJS) $(BUILD_DIR)/model/encoder.o $(BUILD_DIR)/model/layernorm.o $(BUILD_DIR)/model/multi_head_attention.o $(BUILD_DIR)/model/feedforward.o
VIT_OBJS = $(ENC_OBJS) $(BUILD_DIR)/model/vit.o $(BUILD_DIR)/model/patch_embedding.o $(BUILD_DIR)/model/vit.o


# --- Reglas de Compilación ---

# La regla por defecto: si solo escribes "make", se ejecutará esto.
# Construye todos los ejemplos.
all: layernorm multihead feedforward encoder vit

# Regla genérica para compilar cualquier archivo .cpp en un .o
# Make es lo suficientemente inteligente para usar esta regla para todos los MODEL_OBJECTS.
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@) # Crea el directorio de build si no existe (ej. build/core/)
	@echo "Compilando $< -> $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Reglas para construir cada ejecutable final
layernorm: $(LN_OBJS)
	@echo "Linkeando para crear el ejecutable de LayerNorm..."
	$(CXX) $(CXXFLAGS) $(EXAMPLE_DIR)/example_layernorm.cpp $^ -o $(BUILD_DIR)/layernorm.out

multihead: $(MHA_OBJS)
	@echo "Linkeando para crear el ejecutable de Multihead Attention..."
	$(CXX) $(CXXFLAGS) $(EXAMPLE_DIR)/example_multihead_attention.cpp $^ -o $(BUILD_DIR)/multihead.out

feedforward: $(FF_OBJS)
	@echo "Linkeando para crear el ejecutable de FeedForward..."
	$(CXX) $(CXXFLAGS) $(EXAMPLE_DIR)/example_feedforward.cpp $^ -o $(BUILD_DIR)/feedforward.out

encoder: $(ENC_OBJS)
	@echo "Linkeando para crear el ejecutable del Encoder..."
	$(CXX) $(CXXFLAGS) $(EXAMPLE_DIR)/example_encoder.cpp $^ -o $(BUILD_DIR)/encoder.out

vit: $(VIT_OBJS)
	@echo "Linkeando para crear el ejecutable del vit..."
	$(CXX) $(CXXFLAGS) $(EXAMPLE_DIR)/example_vit.cpp $^ -o $(BUILD_DIR)/vit.out

# Regla para limpiar todo lo compilado
clean:
	@echo "Limpiando archivos de compilación..."
	@rm -rf $(BUILD_DIR)

# Le decimos a make que "all" y "clean" no son archivos, sino nombres de comandos.
.PHONY: all clean layernorm multihead feedforward encoder vit