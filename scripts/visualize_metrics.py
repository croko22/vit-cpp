import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import sys

def visualizar_entrenamiento(csv_path):
    """
    Visualiza las métricas de entrenamiento del Vision Transformer
    """
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    plt.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Val Loss', marker='s', markersize=4)
    plt.title('Loss Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(df['epoch'], df['train_accuracy']*100, 'g-', linewidth=2, label='Train Acc', marker='o', markersize=4)
    plt.plot(df['epoch'], df['val_accuracy']*100, 'orange', linewidth=2, label='Val Acc', marker='s', markersize=4)
    plt.title('Accuracy Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. F1-Score curves
    ax3 = plt.subplot(2, 3, 3)
    plt.plot(df['epoch'], df['train_f1_score']*100, 'purple', linewidth=2, label='Train F1', marker='o', markersize=4)
    plt.plot(df['epoch'], df['val_f1_score']*100, 'brown', linewidth=2, label='Val F1', marker='s', markersize=4)
    plt.title('F1-Score Evolution', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Learning Rate y Duration
    ax4 = plt.subplot(2, 3, 4)
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(df['epoch'], df['learning_rate'], 'cyan', linewidth=2, label='Learning Rate', marker='D', markersize=4)
    line2 = ax4_twin.plot(df['epoch'], df['duration_sec'], 'magenta', linewidth=2, label='Duration (s)', marker='^', markersize=4)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate', color='cyan')
    ax4_twin.set_ylabel('Duration (seconds)', color='magenta')
    ax4.set_title('Training Dynamics', fontsize=14, fontweight='bold')
    
    # Combinar leyendas
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Overfitting Analysis
    ax5 = plt.subplot(2, 3, 5)
    overfitting = df['train_accuracy'] - df['val_accuracy']
    plt.plot(df['epoch'], overfitting*100, 'red', linewidth=2, label='Train - Val Acc', marker='x', markersize=5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Overfitting Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Métricas finales y estadísticas
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Estadísticas finales
    final_epoch = df.iloc[-1]
    best_val_acc = df['val_accuracy'].max()
    best_val_epoch = df.loc[df['val_accuracy'].idxmax(), 'epoch']
    avg_duration = df['duration_sec'].mean()
    
    stats_text = f"""
ESTADISTICAS FINALES
{'='*25}

Epoca Final: {int(final_epoch['epoch'])}
Train Acc: {final_epoch['train_accuracy']*100:.2f}%
Val Acc: {final_epoch['val_accuracy']*100:.2f}%
F1-Score: {final_epoch['val_f1_score']*100:.2f}%
Final Loss: {final_epoch['val_loss']:.4f}

MEJORES RESULTADOS
{'='*25}

Mejor Val Acc: {best_val_acc*100:.2f}%
En epoca: {int(best_val_epoch)}
Tiempo promedio/epoca: {avg_duration:.1f}s
Tiempo total: {df['duration_sec'].sum()/60:.1f} min

CONFIGURACION
{'='*25}

Learning Rate: {final_epoch['learning_rate']}
Total Epocas: {len(df)}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = csv_path.replace('.csv', '_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafico guardado en: {output_path}")
    
    plt.show()
    
    return df

def visualizar_comparacion_multiple(dataset_dir):
    """
    Compara múltiples experimentos de un dataset específico
    """
    
    csv_files = glob.glob(f"{dataset_dir}/*.csv")
    
    if len(csv_files) <= 1:
        print("Solo hay un archivo CSV, no se puede hacer comparacion")
        return
    
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Comparar accuracy de validación
    plt.subplot(2, 2, 1)
    for i, csv_file in enumerate(csv_files[:6]):  # Máximo 6 experimentos
        df = pd.read_csv(csv_file)
        filename = Path(csv_file).stem
        plt.plot(df['epoch'], df['val_accuracy']*100, 
                color=colors[i % len(colors)], linewidth=2, 
                label=filename[-8:], marker='o', markersize=3)
    
    plt.title('Comparacion Accuracy Validacion', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparar loss de validación
    plt.subplot(2, 2, 2)
    for i, csv_file in enumerate(csv_files[:6]):
        df = pd.read_csv(csv_file)
        filename = Path(csv_file).stem
        plt.plot(df['epoch'], df['val_loss'], 
                color=colors[i % len(colors)], linewidth=2, 
                label=filename[-8:], marker='s', markersize=3)
    
    plt.title('Comparacion Loss Validacion', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparar F1-Score
    plt.subplot(2, 2, 3)
    for i, csv_file in enumerate(csv_files[:6]):
        df = pd.read_csv(csv_file)
        filename = Path(csv_file).stem
        plt.plot(df['epoch'], df['val_f1_score']*100, 
                color=colors[i % len(colors)], linewidth=2, 
                label=filename[-8:], marker='^', markersize=3)
    
    plt.title('Comparacion F1-Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1-Score (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Tiempo de entrenamiento
    plt.subplot(2, 2, 4)
    exp_names = []
    avg_times = []
    
    for csv_file in csv_files[:6]:
        df = pd.read_csv(csv_file)
        filename = Path(csv_file).stem
        exp_names.append(filename[-8:])  # Últimos 8 caracteres del nombre
        avg_times.append(df['duration_sec'].mean())
    
    bars = plt.bar(exp_names, avg_times, color=colors[:len(exp_names)])
    plt.title('Tiempo Promedio por Epoca', fontsize=14, fontweight='bold')
    plt.xlabel('Experimento')
    plt.ylabel('Tiempo (segundos)')
    plt.xticks(rotation=45)
    
    # Agregar valores encima de las barras
    for bar, time in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time:.1f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/comparison_visualization.png", dpi=300, bbox_inches='tight')
    print(f"Comparacion guardada en: {dataset_dir}/comparison_visualization.png")
    plt.show()

def main():
    """Función principal"""
    if len(sys.argv) != 2:
        print("Uso: python visualize_metrics.py <dataset_name>")
        print("Ejemplo: python visualize_metrics.py mnist")
        print("         python visualize_metrics.py fashionmnist")
        print("         python visualize_metrics.py bloodmnist")
        return
    
    dataset_name = sys.argv[1]
    logs_dir = f"../logs/{dataset_name}"
    
    print(f"Script de Visualizacion ViT Training - {dataset_name.upper()}")
    print("="*50)
    
    # Verificar que el directorio existe
    if not Path(logs_dir).exists():
        print(f"Error: No existe el directorio {logs_dir}")
        return
    
    # Buscar archivos CSV en el directorio del dataset
    csv_files = glob.glob(f"{logs_dir}/*.csv")
    
    if not csv_files:
        print(f"No se encontraron archivos CSV en {logs_dir}")
        return
    
    print(f"Archivos encontrados: {len(csv_files)}")
    for i, file in enumerate(csv_files):
        print(f"  {i+1}. {Path(file).name}")
    
    # Visualizar el más reciente
    latest_file = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"\nVisualizando archivo mas reciente: {Path(latest_file).name}")
    
    try:
        df = visualizar_entrenamiento(latest_file)
        
        # Si hay múltiples archivos, hacer comparación
        if len(csv_files) > 1:
            print(f"\nGenerando comparacion multiple para {dataset_name}...")
            visualizar_comparacion_multiple(logs_dir)
        
        print("\nVisualizacion completada!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()