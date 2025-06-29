# Transformer de traducción

Este proyecto implementa un modelo **Transformer** utilizando únicamente **CUDA y C++**. La implementación es educativa y busca comprender los fundamentos del paper ["Attention is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) a bajo nivel, acelerado en GPU.

## Notebooks
- [TIA - MARIAN | Transformer_Trainer_Notebook](https://colab.research.google.com/drive/134n_xEv7VfA2_5VniJEgzhNcSh8etRPz#scrollTo=VmYGVziz50tu)
- [TIA - MARIAN | input_embeddings](https://colab.research.google.com/drive/12Tq-RRQ8HntnFcKtEui3OznjHztWN92q?usp=sharing)

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

      * `layernorm`
      * `multihead`
      * `feedforward`
      * `encoder`
      * `decoder`
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