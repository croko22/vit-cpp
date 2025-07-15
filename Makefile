CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -Iinclude

# Directorios
SRC_DIR = src
BUILD_DIR = build
EXAMPLES_DIR = examples
APP_DIR = app

# Archivos fuente automáticos
CORE_SOURCES = $(wildcard $(SRC_DIR)/core/*.cpp)
MODEL_SOURCES = $(wildcard $(SRC_DIR)/model/*.cpp)

# Objetos correspondientes
CORE_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CORE_SOURCES))
MODEL_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(MODEL_SOURCES))
ALL_OBJECTS = $(CORE_OBJECTS) $(MODEL_OBJECTS)

# Ejecutables
EXECUTABLES = tensor_test layernorm multihead feedforward encoder vit patch_embedding train

# --- REGLAS PRINCIPALES ---

all: $(EXECUTABLES)

# Regla genérica para compilar .cpp a .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "🔨 Compilando $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# --- EJECUTABLES INDIVIDUALES ---

tensor_test: $(BUILD_DIR)/core/tensor.o
	@echo "🔗 Linkeando Tensor Test"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_tensor.cpp $^ -o $(BUILD_DIR)/$@.out

layernorm: $(BUILD_DIR)/core/tensor.o $(BUILD_DIR)/model/layernorm.o
	@echo "🔗 Linkeando LayerNorm"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_layernorm.cpp $^ -o $(BUILD_DIR)/$@.out

multihead: $(BUILD_DIR)/core/tensor.o $(BUILD_DIR)/model/multi_head_attention.o
	@echo "🔗 Linkeando MultiHead Attention"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_multihead_attention.cpp $^ -o $(BUILD_DIR)/$@.out

feedforward: $(BUILD_DIR)/core/tensor.o $(BUILD_DIR)/model/feedforward.o
	@echo "🔗 Linkeando FeedForward"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_feedforward.cpp $^ -o $(BUILD_DIR)/$@.out

encoder: $(BUILD_DIR)/core/tensor.o \
		 $(BUILD_DIR)/model/layernorm.o \
		 $(BUILD_DIR)/model/multi_head_attention.o \
		 $(BUILD_DIR)/model/feedforward.o \
		 $(BUILD_DIR)/model/encoder.o
	@echo "🔗 Linkeando Encoder"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_encoder.cpp $^ -o $(BUILD_DIR)/$@.out

patch_embedding: $(BUILD_DIR)/core/tensor.o $(BUILD_DIR)/model/patch_embedding.o
	@echo "🔗 Linkeando Patch Embedding"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_patch_embedding.cpp $^ -o $(BUILD_DIR)/$@.out

vit: $(BUILD_DIR)/core/tensor.o \
	 $(BUILD_DIR)/model/layernorm.o \
	 $(BUILD_DIR)/model/multi_head_attention.o \
	 $(BUILD_DIR)/model/feedforward.o \
	 $(BUILD_DIR)/model/encoder.o \
	 $(BUILD_DIR)/model/patch_embedding.o \
	 $(BUILD_DIR)/model/vit.o
	@echo "🔗 Linkeando Vision Transformer"
	@$(CXX) $(CXXFLAGS) $(EXAMPLES_DIR)/example_vit.cpp $^ -o $(BUILD_DIR)/$@.out

train: $(BUILD_DIR)/core/activation.o \
	   $(BUILD_DIR)/core/random.o \
	   $(BUILD_DIR)/core/tensor.o \
	   $(BUILD_DIR)/model/vit.o \
	   $(BUILD_DIR)/model/encoder.o \
	   $(BUILD_DIR)/model/layernorm.o \
	   $(BUILD_DIR)/model/linear.o \
	   $(BUILD_DIR)/model/mlp.o
	@echo "🔗 Linkeando Training App"
	@$(CXX) $(CXXFLAGS) $(APP_DIR)/train.cpp $^ -o $(BUILD_DIR)/$@.out

infer: $(BUILD_DIR)/core/activation.o \
	   $(BUILD_DIR)/core/random.o \
	   $(BUILD_DIR)/core/tensor.o \
	   $(BUILD_DIR)/model/vit.o \
	   $(BUILD_DIR)/model/encoder.o \
	   $(BUILD_DIR)/model/layernorm.o \
	   $(BUILD_DIR)/model/linear.o \
	   $(BUILD_DIR)/model/mlp.o
	@echo "🔗 Linkeando Inference App"
	@$(CXX) $(CXXFLAGS) $(APP_DIR)/infer.cpp $^ -o $(BUILD_DIR)/$@.out


# --- COMANDOS DE EJECUCIÓN ---

run_%: %
	@echo "🚀 Ejecutando $*"
	@./$(BUILD_DIR)/$*.out

# Comandos específicos
run_train: train
	@echo "🏃 Iniciando entrenamiento..."
	@./$(BUILD_DIR)/train.out

run_infer: infer
	@echo "🏃 Iniciando inferencia..."
	@# Se agrupan todos los comandos en un solo bloque de shell para que las variables persistan.
	@MODEL=$$(ls models/*.bin 2>/dev/null | head -1); \
	IMAGE=$$(ls data/predict/*.csv 2>/dev/null | head -1); \
	if [ -z "$$MODEL" ] || [ -z "$$IMAGE" ]; then \
		echo "❌ Error: No se encontró un modelo (.bin) en la carpeta 'models/' o una imagen (.csv) en 'data/predict/'"; \
		exit 1; \
	fi; \
	echo "   Modelo a usar: $$MODEL"; \
	echo "   Imagen a usar: $$IMAGE"; \
	echo "------------------------------------------"; \
	echo "Ejecutando el programa de inferencia..."; \
	./$(BUILD_DIR)/infer.out "$$MODEL" "$$IMAGE"

# Comando para hacer predicción completa
predict: infer
	@echo "🎯 Predicción completa:"
	@echo "1. Extrayendo imagen aleatoria..."
	@python3 scripts/extract_image.py data/mnist/mnist_test.csv -o data/predict -n 1
	@echo "2. Ejecutando inferencia..."
	@MODEL=$$(ls models/*.bin | head -1); \
	IMAGE=$$(ls data/predict/*.csv | head -1); \
	if [ -n "$$MODEL" ] && [ -n "$$IMAGE" ]; then \
		./$(BUILD_DIR)/infer.out "$$MODEL" "$$IMAGE"; \
	else \
		echo "❌ No se encontró modelo o imagen"; \
	fi

# --- UTILIDADES ---

# Compilar solo los objetos
compile: $(ALL_OBJECTS)
	@echo "✅ Todos los objetos compilados"

# Mostrar información
info:
	@echo "📊 Información del proyecto:"
	@echo "  Core sources: $(words $(CORE_SOURCES)) archivos"
	@echo "  Model sources: $(words $(MODEL_SOURCES)) archivos"
	@echo "  Total objects: $(words $(ALL_OBJECTS)) archivos"
	@echo "  Ejecutables: $(EXECUTABLES)"

# Verificar archivos faltantes
check:
	@echo "🔍 Verificando archivos..."
	@for exe in $(EXECUTABLES); do \
		if [ -f "$(EXAMPLES_DIR)/example_$$exe.cpp" ] || [ -f "$(APP_DIR)/$$exe.cpp" ]; then \
			echo "✅ $$exe: archivo fuente encontrado"; \
		else \
			echo "❌ $$exe: archivo fuente FALTANTE"; \
		fi \
	done

# Limpiar
clean:
	@echo "🧹 Limpiando..."
	@rm -rf $(BUILD_DIR)

# Limpiar y recompilar todo
rebuild: clean all

# --- AYUDA ---

help:
	@echo "🛠️  Makefile para Marian - Vision Transformer"
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make              - Compilar todo"
	@echo "  make <target>     - Compilar target específico"
	@echo "  make run_<target> - Compilar y ejecutar target"
	@echo ""
	@echo "Targets disponibles:"
	@echo "  $(EXECUTABLES)"
	@echo ""
	@echo "Utilidades:"
	@echo "  make compile      - Solo compilar objetos"
	@echo "  make clean        - Limpiar archivos de build"
	@echo "  make rebuild      - Limpiar y recompilar todo"
	@echo "  make info         - Mostrar información del proyecto"
	@echo "  make check        - Verificar archivos faltantes"
	@echo "  make help         - Mostrar esta ayuda"

.PHONY: $(EXECUTABLES) all clean compile rebuild info check help run_% run_train

# Evitar borrar archivos intermedios
.PRECIOUS: $(BUILD_DIR)/%.o