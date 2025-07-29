import numpy as np
import pandas as pd
import os

def bloodmnist_npy_to_csv(data_path="../data/bloodmnist/", save_path="./", grayscale=True):
    """
    Convierte los archivos .npy de BloodMNIST a formato CSV
    
    Args:
        data_path (str): Ruta donde están los archivos .npy
        save_path (str): Ruta donde guardar los archivos CSV
        grayscale (bool): Si convertir a grayscale o mantener RGB
    """
    
    print("Cargando archivos .npy de BloodMNIST...")
    
    def load_and_convert_split(split_name):
        """Carga un split y lo convierte a CSV"""
        
        print(f"Procesando {split_name} dataset...")
        
        images_file = os.path.join(data_path, f"{split_name}_images.npy")
        labels_file = os.path.join(data_path, f"{split_name}_labels.npy")
        
        if not os.path.exists(images_file) or not os.path.exists(labels_file):
            print(f"Archivos no encontrados para {split_name}")
            return None
        
        images = np.load(images_file)
        labels = np.load(labels_file)
        
        print(f"  - Forma de imagenes: {images.shape}")
        print(f"  - Forma de etiquetas: {labels.shape}")
        
        if len(images.shape) == 4:
            num_samples = images.shape[0]
            if grayscale:
                images_gray = np.mean(images, axis=3)
                images_flat = images_gray.reshape(num_samples, -1)
                print(f"  - Convertido a grayscale: {images_gray.shape}")
            else:
                images_flat = images.reshape(num_samples, -1)
        else:
            images_flat = images
        
        labels_flat = labels.flatten()
        
        print(f"  - Imagenes aplanadas: {images_flat.shape}")
        print(f"  - Etiquetas aplanadas: {labels_flat.shape}")
        
        num_pixels = images_flat.shape[1]
        pixel_columns = [f'pixel{i+1}' for i in range(num_pixels)]
        columns = ['label'] + pixel_columns
        
        data = np.column_stack([labels_flat, images_flat])
        df = pd.DataFrame(data, columns=columns)
        
        df['label'] = df['label'].astype(int)
        
        suffix = "_grayscale" if grayscale else ""
        csv_filename = os.path.join(save_path, f'bloodmnist_{split_name}{suffix}.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"Dataset {split_name} guardado en: {csv_filename}")
        print(f"  - Forma del CSV: {df.shape}")
        print(f"  - Clases unicas: {sorted(df['label'].unique())}")
        
        return df
    
    splits = ['train', 'val', 'test']
    dataframes = {}
    
    for split in splits:
        df = load_and_convert_split(split)
        if df is not None:
            dataframes[split] = df
    
    if len(dataframes) == 3:
        print("\nCreando dataset combinado...")
        combined_df = pd.concat(list(dataframes.values()), ignore_index=True)
        suffix = "_grayscale" if grayscale else ""
        combined_filename = os.path.join(save_path, f'bloodmnist_combined{suffix}.csv')
        combined_df.to_csv(combined_filename, index=False)
        
        print(f"Dataset combinado guardado en: {combined_filename}")
        print(f"  - Forma total: {combined_df.shape}")
        
        print(f"\nDistribucion de clases en el dataset completo:")
        class_counts = combined_df['label'].value_counts().sort_index()
        for class_id, count in class_counts.items():
            print(f"  Clase {class_id}: {count} muestras")
        
        return dataframes['train'], dataframes['val'], dataframes['test'], combined_df
    
    return dataframes.get('train'), dataframes.get('val'), dataframes.get('test'), None

def main():
    """Función principal"""
    print("Script para convertir BloodMNIST (.npy) a CSV")
    print("="*45)
    
    output_dir = "../data/bloodmnist_csv"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        train_df, val_df, test_df, combined_df = bloodmnist_npy_to_csv(
            data_path="../data/bloodmnist/",
            save_path=output_dir,
            grayscale=True
        )
        
        print(f"\nConversion completada exitosamente!")
        print(f"Archivos guardados en: {output_dir}")
        
        if combined_df is not None:
            print(f"\nPreview del dataset:")
            print(combined_df.head())
            print(f"\nInfo del dataset:")
            print(f"  - Total de muestras: {len(combined_df)}")
            print(f"  - Numero de caracteristicas: {len(combined_df.columns) - 1}")
            print(f"  - Rango de valores de pixeles: [{combined_df.iloc[:, 1:].min().min()}, {combined_df.iloc[:, 1:].max().max()}]")
        
    except Exception as e:
        print(f"Error durante la conversion: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()