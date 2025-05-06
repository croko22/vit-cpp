# Nombre del ejecutable
TARGET = attention

# Archivos fuente
SRC = main.cu attention.cu matmul.cu softmax.cu

# Compilador CUDA
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -arch=sm_75 -std=c++17

# Regla principal
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# Limpieza de archivos generados
clean:
	rm -f $(TARGET)