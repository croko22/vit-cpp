# Vision Transformer

Este proyecto implementa un modelo **Vision Transformer (ViT)** utilizando únicamente **CUDA y C++**. La implementación es educativa y busca comprender los fundamentos del paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)](https://arxiv.org/pdf/2010.11929).

A diferencia de las redes neuronales convolucionales tradicionales, este enfoque transforma una imagen en pequeños *patches* que son procesados como secuencias, permitiendo el uso de la arquitectura Transformer originalmente diseñada para tareas de procesamiento de lenguaje natural.

## Objetivo

El objetivo principal es explorar cómo los Transformers pueden aplicarse al reconocimiento visual desde cero, sin depender de frameworks de alto nivel, y acelerando el entrenamiento e inferencia mediante **CUDA** y **programación en bajo nivel con C++**.

## ¿Qué es Vision Transformer?

El **Vision Transformer (ViT)** divide una imagen en bloques fijos (por ejemplo, de 16x16 píxeles), los aplane y los trate como "palabras" o tokens. Estos tokens se procesan mediante una serie de capas Transformer, seguidas por una capa final de clasificación.

### Características clave:
- División de imágenes en *patches*.
- Uso de embeddings aprendidos para cada patch.
- Inserción de un token especial `[CLS]` para la clasificación.
- Uso de atención multi-cabeza y posiciones absolutas aprendidas.

![architecture](docs/img/architecture.png)

## Referencias

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Notebooks
- [ViT - Keras](https://colab.research.google.com/drive/1OlspI87qJouwFWuTzH2k29ai4XsfYnrT?usp=sharing)
- [ViT - Pytorch](https://colab.research.google.com/drive/1J_GLR-PMsMiuiRqsPXpJXCT8LMaOfuq1?usp=sharing)

## Cómo compilar y ejecutar

Este proyecto utiliza un `Makefile` para gestionar la compilación de forma eficiente y un script de ayuda (`run.sh`) para simplificar la ejecución de tareas comunes.

### Requisitos

  * Un compilador de C++ compatible con C++17 (ej. `g++`).
  * La utilidad `make`.

### Método 1: Usando el script de ayuda (`run.sh`) (Recomendado)

Este script proporciona una interfaz sencilla para las operaciones más comunes.

**1. Dar permisos de ejecución al script:**

Primero, asegúrate de que el script sea ejecutable. Este comando solo necesita ser ejecutado una vez.

```bash
chmod +x run.sh
```

**2. Comandos disponibles:**

* **Entrenar el modelo:**
  Entrena un Vision Transformer con los datasets especificados.

  ```bash
  ./run.sh train <archivo_entrenamiento.csv> <archivo_prueba.csv>
  ```

  Ejemplo:
  ```bash
  ./run.sh train data/mnist/mnist_train.csv data/mnist/mnist_test.csv
  ```

* **Hacer inferencia con imagen específica:**
  Realiza predicción sobre una imagen específica usando un modelo entrenado.

  ```bash
  ./run.sh infer <modelo.bin> <imagen.csv>
  ```

  Ejemplo:
  ```bash
  ./run.sh infer models/vit_20250715_161742.bin data/predict/sample_imagen.csv
  ```

* **Predicción automática:**
  Extrae automáticamente una imagen aleatoria del dataset de prueba y realiza predicción.

  ```bash
  ./run.sh predict
  ```

* **Limpiar el proyecto:**
  Elimina la carpeta `build/` y todos los archivos compilados.

  ```bash
  ./run.sh clean
  ```

   g++ examples/test_mha.cpp src/model/multihead_attention.cpp src/core/tensor.cpp src/model/linear.cpp src/core/activ
ation.cpp src/core/random.cpp && ./a.out

# Compilar
g++ -O2 benchmarks/bench_attention.cpp src/model/multihead_attention.cpp src/core/tensor.cpp src/core/activation.cpp src/model/linear.cpp src/core/random.cpp -Iinclude

# Ejecutar
./a.out

## Resultados

## Entrenamiento MNIST SGD
![train](docs/img/train1.png)

### Prediccion mnist
![infer](docs/img/predict1.png)

- ./run.sh train data/mnist/mnist_train.csv data/mnist/mnist_test.csv
- 