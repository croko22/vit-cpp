import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def get_class_names(dataset_name):
    """
    Devuelve la lista de nombres de clases para un dataset específico.
    """
    class_map = {
        'fashion': [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ],
        'mnist': [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ],
        'blood': [
            'basophil', 'eosinophil', 'erythroblast', 'ig',
            'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
        ]
    }
    return class_map.get(dataset_name.lower())

def main():
    """
    Función principal para cargar datos, generar y mostrar la matriz de confusión.
    """
    parser = argparse.ArgumentParser(
        description="Genera una visualización de una matriz de confusión desde un archivo CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--file',
        type=str,
        default='confusion_matrix.csv',
        help="Ruta al archivo CSV de la matriz de confusión.\n(default: %(default)s)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['mnist', 'fashion', 'blood'],
        help="Nombre del dataset para usar las etiquetas de clase correctas.\n(opciones: %(choices)s)"
    )

    args = parser.parse_args()

    class_names = get_class_names(args.dataset)
    if not class_names:
        print(f"Error: Nombre de dataset '{args.dataset}' no reconocido.")
        sys.exit(1)

    try:
        cm_df = pd.read_csv(args.file, header=None)
        confusion_matrix = cm_df.to_numpy()
        print(f"Matriz de confusión cargada desde '{args.file}'.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{args.file}'.")
        sys.exit(1)

    if confusion_matrix.shape[0] != len(class_names) or confusion_matrix.shape[1] != len(class_names):
        print(f"Error: La dimensión de la matriz ({confusion_matrix.shape[0]}x{confusion_matrix.shape[1]}) no coincide con el número de clases para '{args.dataset}' ({len(class_names)}).")
        sys.exit(1)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=.5,
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names
    )

    ax.set_title(f'Matriz de Confusión - {args.dataset.capitalize()}', fontsize=18, pad=20)
    ax.set_xlabel('Etiqueta Predicha', fontsize=14)
    ax.set_ylabel('Etiqueta Real', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
