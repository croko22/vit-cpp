import numpy as np
import pandas as pd
import os

def bloodmnist_npy_to_csv(data_path="../data/bloodmnist/", save_path="./"):
    """
    Convierte los archivos .npy de BloodMNIST a formato CSV
    
    Args:
        data_path (str): Ruta donde están los archivos .npy
        save_path (str): Ruta donde guardar los archivos CSV
    """
    
    print("Cargando archivos .npy de BloodMNIST...")
    
    def load_and_convert_split(split_name):
        """Carga un split y lo convierte a CSV"""
        
        print(f"Procesando {split_name} dataset...")
        
        # Cargar imágenes y etiquetas
        images_file = os.path.join(data_path, f"{split_name}_images.npy")
        labels_file = os.path.join(data_path, f"{split_name}_labels.npy")
        
        if not os.path.exists(images_file) or not os.path.exists(labels_file):
            print(f"❌ Archivos no encontrados para {split_name}")
            return None
        
        images = np.load(images_file)
        labels = np.load(labels_file)
        
        print(f"  - Forma de imágenes: {images.shape}")
        print(f"  - Forma de etiquetas: {labels.shape}")
        
        # Aplanar las imágenes
        # Si las imágenes son (N, H, W, C), las aplanamos a (N, H*W*C)
        if len(images.shape) == 4:
            num_samples = images.shape[0]
            images_flat = images.reshape(num_samples, -1)
        else:
            images_flat = images
        
        # Aplanar etiquetas si es necesario
        labels_flat = labels.flatten()
        
        print(f"  - Imágenes aplanadas: {images_flat.shape}")
        print(f"  - Etiquetas aplanadas: {labels_flat.shape}")
        
        # Crear nombres de columnas
        num_pixels = images_flat.shape[1]
        pixel_columns = [f'pixel{i+1}' for i in range(num_pixels)]
        columns = ['label'] + pixel_columns
        
        # Crear DataFrame
        data = np.column_stack([labels_flat, images_flat])
        df = pd.DataFrame(data, columns=columns)
        
        # Convertir label a int
        df['label'] = df['label'].astype(int)
        
        # Guardar CSV
        csv_filename = os.path.join(save_path, f'bloodmnist_{split_name}.csv')
        df.to_csv(csv_filename, index=False)
        
        print(f"✓ {split_name} dataset guardado en: {csv_filename}")
        print(f"  - Forma del CSV: {df.shape}")
        print(f"  - Clases únicas: {sorted(df['label'].unique())}")
        
        return df
    
    # Procesar cada split
    splits = ['train', 'val', 'test']
    dataframes = {}
    
    for split in splits:
        df = load_and_convert_split(split)
        if df is not None:
            dataframes[split] = df
    
    # Crear dataset combinado si tenemos todos los splits
    if len(dataframes) == 3:
        print("\nCreando dataset combinado...")
        combined_df = pd.concat(list(dataframes.values()), ignore_index=True)
        combined_filename = os.path.join(save_path, 'bloodmnist_combined.csv')
        combined_df.to_csv(combined_filename, index=False)
        
        print(f"✓ Dataset combinado guardado en: {combined_filename}")
        print(f"  - Forma total: {combined_df.shape}")
        
        # Mostrar distribución de clases
        print(f"\nDistribución de clases en el dataset completo:")
        class_counts = combined_df['label'].value_counts().sort_index()
        for class_id, count in class_counts.items():
            print(f"  Clase {class_id}: {count} muestras")
        
        return dataframes['train'], dataframes['val'], dataframes['test'], combined_df
    
    return dataframes.get('train'), dataframes.get('val'), dataframes.get('test'), None

def main():
    """Función principal"""
    print("Script para convertir BloodMNIST (.npy) a CSV")
    print("="*45)
    
    # Crear directorio de salida si no existe
    output_dir = "../data/bloodmnist_csv"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Ejecutar conversión
        train_df, val_df, test_df, combined_df = bloodmnist_npy_to_csv(
            data_path="../data/bloodmnist/",
            save_path=output_dir
        )
        
        print(f"\n✅ ¡Conversión completada exitosamente!")
        print(f"📁 Archivos guardados en: {output_dir}")
        
        # Mostrar preview de los datos si tenemos el dataset combinado
        if combined_df is not None:
            print(f"\n📊 Preview del dataset:")
            print(combined_df.head())
            print(f"\n📊 Info del dataset:")
            print(f"  - Total de muestras: {len(combined_df)}")
            print(f"  - Número de características: {len(combined_df.columns) - 1}")
            print(f"  - Rango de valores de píxeles: [{combined_df.iloc[:, 1:].min().min()}, {combined_df.iloc[:, 1:].max().max()}]")
        
    except Exception as e:
        print(f"❌ Error durante la conversión: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()