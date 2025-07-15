#!/bin/bash
# filepath: run.sh

BUILD_DIR="build"

# Función de ayuda
show_help() {
    echo "Uso: ./run.sh <comando> [argumentos...]"
    echo ""
    echo "Comandos disponibles:"
    echo "  train <train.csv> <test.csv>     - Entrenar modelo"
    echo "  infer <modelo.bin> <imagen.csv>  - Hacer inferencia"
    echo "  predict                          - Extraer imagen y predecir"
    echo "  clean                            - Limpiar archivos build"
    echo ""
    echo "Ejemplos:"
    echo "  ./run.sh train data/mnist/mnist_train.csv data/mnist/mnist_test.csv"
    echo "  ./run.sh infer models/modelo.bin data/predict/imagen.csv"
    echo "  ./run.sh predict"
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
        if [ $# -ne 2 ]; then
            echo "Error: train requiere 2 argumentos"
            echo "Uso: ./run.sh train <train.csv> <test.csv>"
            exit 1
        fi
        
        echo "Compilando entrenamiento..."
        make train
        
        if [ $? -eq 0 ]; then
            echo "Ejecutando entrenamiento..."
            ./${BUILD_DIR}/train.out "$1" "$2"
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
        
        echo "Compilando inferencia..."
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
        echo "Compilando inferencia..."
        make infer
        
        if [ $? -ne 0 ]; then
            echo "Error en compilación"
            exit 1
        fi
        
        echo "Extrayendo imagen aleatoria..."
        python3 scripts/extract_image.py data/mnist/mnist_test.csv
        
        MODEL=$(ls models/*.bin 2>/dev/null | head -1)
        IMAGE=$(ls data/predict/*_raw.csv 2>/dev/null | head -1)
        
        if [ -z "$MODEL" ] || [ -z "$IMAGE" ]; then
            echo "Error: No se encontró modelo o imagen"
            exit 1
        fi
        
        echo "Ejecutando inferencia..."
        ./${BUILD_DIR}/infer.out "$MODEL" "$IMAGE"
        ;;
        
    "clean")
        echo "Limpiando archivos build..."
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