# --- Compilador y Flags ---
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude

# --- Directorios ---
SRC_DIR = src
BUILD_DIR = build
APP_DIR = app

# --- Detección Automática de Archivos Fuente (.cpp) ---
# Encuentra todos los archivos .cpp en los subdirectorios de src
SOURCES = $(wildcard $(SRC_DIR)/core/*.cpp) \
          $(wildcard $(SRC_DIR)/model/*.cpp) \
          $(wildcard $(SRC_DIR)/optimizer/*.cpp)

# --- Generación Automática de Archivos Objeto (.o) ---
# Convierte la lista de fuentes a su ruta de objeto correspondiente en build/
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# --- Reglas Principales ---
all: train infer

# Regla para compilar el programa de entrenamiento
# Enlaza el archivo train.cpp con TODOS los objetos compilados
train: $(OBJECTS)
	@echo "Enlazando programa de entrenamiento..."
	$(CXX) $(CXXFLAGS) $(APP_DIR)/train.cpp $(OBJECTS) -o $(BUILD_DIR)/train.out

# Regla para compilar el programa de inferencia
# Enlaza el archivo infer.cpp con TODOS los objetos compilados
infer: $(OBJECTS)
	@echo "Enlazando programa de inferencia..."
	$(CXX) $(CXXFLAGS) $(APP_DIR)/infer.cpp $(OBJECTS) -o $(BUILD_DIR)/infer.out

# --- Regla Genérica para Compilar .cpp a .o ---
# Esta regla sabe cómo compilar CUALQUIER archivo .cpp de src/ a build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compilando $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Limpieza ---
clean:
	@echo "Limpiando directorio de compilación..."
	rm -rf $(BUILD_DIR)

# --- Evitar conflictos con nombres de archivos ---
.PHONY: all train infer clean