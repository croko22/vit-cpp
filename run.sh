#!/bin/bash
# filepath: run.sh

BUILD_DIR="build"

# Función de ayuda
show_help() {
    echo "Uso: ./run.sh <comando> [argumentos...]"
    echo ""
    echo "Comandos disponibles:"
    echo "  train <config.cfg>               - Entrenar un modelo usando un archivo de configuración."
    echo "  evaluate <modelo.bin> <dataset>  - Evaluar un modelo y generar matriz de confusión."
    echo "  predict [dataset]                - Extraer imagen aleatoria de un dataset y predecir."
    echo "  infer <modelo.bin> <imagen.csv>    - Hacer inferencia directa con un archivo de imagen."
    echo "  clean                              - Limpiar archivos de compilación."
    echo ""
    echo "Ejemplos:"
    echo "  ./run.sh train configs/mnist.cfg"
    echo "  ./run.sh evaluate models/mnist/vit_..._acc_98.00.bin mnist"
    echo "  ./run.sh predict fashionmnist"
}

# Verificar argumentos
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

case $COMMAND in
    "train")
        mkdir -p logs/mnist
        
        if [ $# -ne 1 ]; then
            echo "Error: train requiere 1 argumento"
            echo "Uso: ./run.sh train <ruta_a_config.cfg>"
            exit 1
        fi
        
        echo "Compilando programa de entrenamiento..."
        make train
        
        if [ $? -eq 0 ]; then
            echo "Iniciando entrenamiento con configuración: $1"
            ./${BUILD_DIR}/train.out "$1"
        else
            echo "Error en compilación"
            exit 1
        fi
        ;;

    "evaluate")
        if [ $# -ne 2 ]; then
            echo "Error: evaluate requiere 2 argumentos"
            echo "Uso: ./run.sh evaluate <modelo.bin> <dataset_name>"
            exit 1
        fi

        MODEL_PATH=$1
        DATASET_NAME=$2
        TEST_CSV_PATH=""

        if [ "$DATASET_NAME" == "mnist" ]; then
            TEST_CSV_PATH="data/mnist/mnist_test.csv"
        elif [ "$DATASET_NAME" == "fashionmnist" ]; then
            TEST_CSV_PATH="data/fashion_mnist/fashion-mnist_test.csv"
        elif [ "$DATASET_NAME" == "bloodmnist" ]; then
            TEST_CSV_PATH="data/bloodmnist_csv/bloodmnist_test.csv"
        else
            echo "Error: Dataset '$DATASET_NAME' no reconocido."
            exit 1
        fi

        echo "Compilando programa de evaluación..."
        make evaluate

        if [ $? -eq 0 ]; then
            echo "Ejecutando evaluación para el modelo: $MODEL_PATH"
            ./${BUILD_DIR}/evaluate.out "$MODEL_PATH" "$TEST_CSV_PATH" "$DATASET_NAME"
        else
            echo "Error en compilación"
            exit 1
        fi
        ;;
        
    "infer")
        if [ $# -ne 2 ]; then
            echo "Error: infer requiere 2 argumentos"
            echo "Uso: ./run.sh infer <modelo.bin> <imagen.csv>"
            exit 1
        fi
        
        echo "Compilando programa de inferencia..."
        make infer
        
        if [ $? -eq 0 ]; then
            echo "Ejecutando inferencia..."
            ./${BUILD_DIR}/infer.out "$1" "$2"
        else
            echo "Error en compilación"
            exit 1
        fi
        ;;
        
    "predict")
        DATASET_NAME=${1:-mnist} # Usa 'mnist' si no se proporciona argumento
        TEST_CSV_PATH=""

        if [ "$DATASET_NAME" == "mnist" ]; then
            TEST_CSV_PATH="data/mnist/mnist_test.csv"
        elif [ "$DATASET_NAME" == "fashionmnist" ]; then
            TEST_CSV_PATH="data/fashion_mnist/fashion-mnist_test.csv"
        elif [ "$DATASET_NAME" == "bloodmnist" ]; then
            TEST_CSV_PATH="data/bloodmnist_csv/bloodmnist_test.csv"
        else
            echo "Error: Dataset '$DATASET_NAME' no reconocido."
            exit 1
        fi

        echo "Compilando programa de inferencia..."
        make infer
        
        if [ $? -ne 0 ]; then
            echo "Error en compilación"
            exit 1
        fi
        
        echo "Extrayendo imagen aleatoria de $DATASET_NAME..."
        python3 scripts/extract_image.py "$TEST_CSV_PATH"
        
        MODEL=$(ls -t models/$DATASET_NAME/*.bin 2>/dev/null | head -1)
        IMAGE=$(ls -t data/predict/*_raw.csv 2>/dev/null | head -1)
        
        if [ -z "$MODEL" ]; then
            echo "Error: No se encontró ningún modelo en el directorio models/$DATASET_NAME/"
            exit 1
        fi
        
        if [ -z "$IMAGE" ]; then
            echo "Error: No se encontró imagen para predecir en data/predict/"
            exit 1
        fi
        
        echo "Usando modelo más reciente: $MODEL"
        echo "Ejecutando inferencia en imagen: $IMAGE"
        ./${BUILD_DIR}/infer.out "$MODEL" "$IMAGE"
        ;;
        
    "clean")
        echo "Limpiando archivos de compilación..."
        make clean
        ;;
        
    "help"|"-h"|"--help")
        show_help
        ;;
        
    *)
        echo "Comando desconocido: $COMMAND"
        show_help
        exit 1
        ;;
esac
