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

<!-- - [TIA - MARIAN | Transformer_Trainer_Notebook](https://colab.research.google.com/drive/134n_xEv7VfA2_5VniJEgzhNcSh8etRPz#scrollTo=VmYGVziz50tu)
- [TIA - MARIAN | input_embeddings](https://colab.research.google.com/drive/12Tq-RRQ8HntnFcKtEui3OznjHztWN92q?usp=sharing) -->

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

  * **Construir todos los ejemplos:**
    Compila todas las demostraciones de los componentes (LayerNorm, Encoder, Transformer, etc.) y deja los ejecutables en la carpeta `build/`.

    ```bash
    ./run.sh build
    ```

  * **Compilar y ejecutar un ejemplo específico:**
    Este comando compila (si es necesario) y ejecuta el ejemplo que especifiques.

    ```bash
    ./run.sh run <nombre_del_ejemplo>
    ```

    Nombres de ejemplos disponibles:

      * `patch_embedding`
      * `layernorm`
      * `multihead`
      * `feedforward`
      * `encoder`
      * `transformer`

    Por ejemplo, para ejecutar la demostración completa del Transformer:

    ```bash
    ./run.sh run transformer
    ```

  * **Probar todo el proyecto:**
    Este comando compila todos los ejemplos y los ejecuta uno por uno en secuencia. Es útil para verificar que todos los componentes funcionan correctamente.

    ```bash
    ./run.sh test
    ```

  * **Limpiar el proyecto:**
    Elimina la carpeta `build/` y todos los archivos compilados.

    ```bash
    ./run.sh clean
    ```

  g++ examples/example_patch_embedding.cpp src/core/tensor.cpp src/model/patch_embedding.cpp src/core/ops.cpp && ./a.
out
 g++ examples/example_train_mnist.cpp src/model/vit.cpp src/core/loss.cpp src/core/optimizer.cpp src/core/
tensor.cpp src/core/ops.cpp src/model/encoder.cpp src/model/feedforward.cpp  src/model/layernorm.cpp  src/model/multi_head_attention.cpp src/model/patc
h_embedding.cpp && ./a.out