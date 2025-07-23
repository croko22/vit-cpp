#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include <cassert>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <chrono>
// Assuming these are your project's header files
#include "../include/core/random.h"
#include "../include/core/tensor.h"
#include "../include/core/activation.h"
#include "../include/model/linear.h"
#include "../include/model/layernorm.h"
#include "../include/model/mlp.h"
#include "../include/model/encoder.h"
#include "../include/model/vit.h"

using namespace std;

// DataLoader now handles loading from a single file.
class DataLoader
{
public:
    static pair<vector<Tensor>, vector<int>> load_data(const string &filename, int max_samples_to_load = -1, int num_classes_to_load = 10)
    {
        vector<Tensor> images;
        vector<int> labels;

        ifstream file(filename);
        if (!file.is_open())
        {
            cerr << "Error: No se pudo abrir el archivo " << filename << endl;
            exit(1);
        }

        string line;
        int samples_loaded = 0;

        // Skip header
        getline(file, line);
        while (getline(file, line) && (max_samples_to_load == -1 || samples_loaded < max_samples_to_load))
        {
            stringstream ss(line);
            string cell;

            if (!getline(ss, cell, ','))
                continue;
            int label = stoi(cell);
            if (label >= num_classes_to_load)
                continue;

            Tensor image(28, 28);
            for (int i = 0; i < 784; i++)
            {
                if (!getline(ss, cell, ','))
                    break;
                image(i / 28, i % 28) = stof(cell) / 255.0f;
            }
            images.push_back(image);
            labels.push_back(label);
            samples_loaded++;
        }
        file.close();
        cout << "Datos cargados: " << samples_loaded << " muestras de " << filename << endl;
        return {images, labels};
    }
};

void printProgressBar(int current, int total, int barWidth = 50)
{
    float progress = (float)current / total;
    int pos = (int)(barWidth * progress);
    cout << "[";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            cout << "=";
        else if (i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
}

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        cerr << "❌ Error: Uso incorrecto." << endl;
        cerr << "   Entrenamiento nuevo: " << argv[0] << " <train.csv> <test.csv>" << endl;
        cerr << "   Continuar entrenamiento: " << argv[0] << " <train.csv> <test.csv> <modelo.bin>" << endl;
        return 1;
    }

    string train_filepath = argv[1];
    string test_filepath = argv[2];
    string pretrained_model_path = (argc == 4) ? argv[3] : "";

    cout << "Vision Transformer con Entrenamiento por Batch" << endl;
    cout << "==============================================" << endl;
    Random::seed(42);

    // --- Hyperparameters ---
    int image_size = 28;
    int patch_size = 4;
    int d_model = 64;
    int num_layers = 2;
    int num_classes = 10;
    float initial_learning_rate = 3e-4f;
    float learning_rate = pretrained_model_path.empty() ? 
                         initial_learning_rate : 
                         initial_learning_rate * 0.1f; //Fine-tuning
    int epochs = 10;
    int batch_size = 128;
    float val_split_ratio = 0.1f; // 10% of training data for validation

    // --- Data Loading ---
    //TO CHANGE: Parametro de maximas imagenes
    cout << "Cargando datos..." << endl;
    auto [all_train_images, all_train_labels] = DataLoader::load_data(train_filepath,6000,num_classes);
    auto [test_images, test_labels] = DataLoader::load_data(test_filepath,1000,num_classes);

    // --- Train/Validation Split ---
    vector<int> indices(all_train_images.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), Random::gen);

    size_t val_size = static_cast<size_t>(all_train_images.size() * val_split_ratio);
    vector<Tensor> train_images, val_images;
    vector<int> train_labels, val_labels;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (i < val_size)
        {
            val_images.push_back(all_train_images[indices[i]]);
            val_labels.push_back(all_train_labels[indices[i]]);
        }
        else
        {
            train_images.push_back(all_train_images[indices[i]]);
            train_labels.push_back(all_train_labels[indices[i]]);
        }
    }

    // --- Model Initialization ---
    VisionTransformer vit(image_size, patch_size, d_model, num_layers, num_classes);

    if (!pretrained_model_path.empty())
    {
        cout << "Cargando modelo pre-entrenado de: " << pretrained_model_path << endl;
        try {
            vit.load_model(pretrained_model_path);
            cout << "Modelo cargado exitosamente" << endl;
            
            // Opcional: Verificar que las dimensiones coincidan
            if (vit.image_size != image_size || vit.patch_size != patch_size || 
                vit.d_model != d_model || vit.num_layers != num_layers ||
                vit.num_classes != num_classes)
            {
                cerr << "  Advertencia: Los hiperparámetros no coinciden con el modelo cargado" << endl;
                cerr << "  Usando los parámetros del modelo cargado:" << endl;
                cerr << "  - Tamaño de imagen: " << vit.image_size << endl;
                cerr << "  - Tamaño de patch: " << vit.patch_size << endl;
                cerr << "  - d_model: " << vit.d_model << endl;
                cerr << "  - Capas: " << vit.num_layers << endl;
                cerr << "  - Clases: " << vit.num_classes << endl;
                
                // Actualizar nuestras variables para consistencia
                image_size = vit.image_size;
                patch_size = vit.patch_size;
                d_model = vit.d_model;
                num_layers = vit.num_layers;
                num_classes = vit.num_classes;
            }
        } 
        catch (const std::exception& e) {
            cerr << "Error al cargar el modelo: " << e.what() << endl;
            cerr << "Inicializando nuevo modelo..." << endl;
        }
    }
    else
    {
        cout << "Inicializando nuevo modelo con pesos aleatorios" << endl;
    }

    cout << "\nConfiguración:" << endl;
    cout << "- Imagen: " << image_size << "x" << image_size << endl;
    cout << "- Patch: " << patch_size << "x" << patch_size << endl;
    cout << "- Patches por imagen: " << vit.num_patches << endl;
    cout << "- Dimensión de embedding (d_model): " << d_model << endl;
    cout << "- Capas Transformer: " << num_layers << endl;
    cout << "- Clases: " << num_classes << endl;
    cout << "- Learning rate: " << learning_rate << endl;
    if (!pretrained_model_path.empty())
    {
        cout << "  (Learning rate reducido para fine-tuning)" << endl;
    }
    cout << "- Épocas: " << epochs << endl;
    cout << "- Batch size: " << batch_size << endl;
    cout << "- Muestras de entrenamiento: " << train_images.size() << endl;
    cout << "- Muestras de validación: " << val_images.size() << endl;
    cout << "- Muestras de prueba: " << test_images.size() << endl
         << endl;

    // // --- Training Loop ---
    // cout << "Entrenando..." << endl;
    // for (int epoch = 0; epoch < epochs; epoch++)
    // {
    //     float train_loss = 0.0f;
    //     int train_correct = 0;
    //     vector<int> train_indices(train_images.size());
    //     iota(train_indices.begin(), train_indices.end(), 0);
    //     shuffle(train_indices.begin(), train_indices.end(), Random::gen);

    //     int batch_count = 0;
    //     int total_batches = ceil((float)train_indices.size() / batch_size);

    //     cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
    //     for (size_t batch_start = 0; batch_start < train_indices.size(); batch_start += batch_size)
    //     {
    //         vit.zero_grad();
    //         size_t batch_end = min(batch_start + batch_size, train_indices.size());
    //         Tensor logits;
    //         // int debug_idx = Random::randint(batch_start, batch_end - 1);

    //         for (size_t i = batch_start; i < batch_end; ++i)
    //         {
    //             int idx = train_indices[i];
    //             logits = vit.forward(train_images[idx]);
                
    //             if (idx == 2323) {  // Solo imprimir una muestra aleatoria
    //                 std::cout << "Logits ejemplo " << idx << ": ";
    //                 for (int j = 0; j < num_classes; ++j) 
    //                     std::cout << logits(0,j) << " ";
    //                 std::cout << std::endl;
    //             }

    //             vit.backward(train_labels[idx]);
    //             train_loss += vit.compute_loss(logits, train_labels[idx]);
    //             if (vit.predict(train_images[idx]) == train_labels[idx])
    //                 train_correct++;
    //         }

    //         vit.update_weights(learning_rate);
    //         batch_count++;
    //         printProgressBar(batch_count, total_batches);
    //     }
    //     cout << endl;

    //     // --- Validation Step ---
    //     float val_loss = 0.0f;
    //     int val_correct = 0;
    //     if (!val_images.empty())
    //     {
    //         for (size_t i = 0; i < val_images.size(); i++)
    //         {
    //             Tensor logits = vit.forward(val_images[i]);
    //             val_loss += vit.compute_loss(logits, val_labels[i]);
    //             if (vit.predict(val_images[i]) == val_labels[i])
    //                 val_correct++;
    //         }
    //     }

    //     float avg_train_loss = train_images.empty() ? 0 : train_loss / train_images.size();
    //     float train_acc = train_images.empty() ? 0 : (float)train_correct / train_images.size();
    //     float avg_val_loss = val_images.empty() ? 0 : val_loss / val_images.size();
    //     float val_acc = val_images.empty() ? 0 : (float)val_correct / val_images.size();

    //     cout << "  Entrenamiento - Pérdida: " << fixed << setprecision(4) << avg_train_loss
    //          << " | Precisión: " << setprecision(2) << train_acc * 100 << "%" << endl;
    //     cout << "  Validación    - Pérdida: " << fixed << setprecision(4) << avg_val_loss
    //          << " | Precisión: " << setprecision(2) << val_acc * 100 << "%" << endl
    //          << endl;
    // }

    // --- Debug: Modelo Lineal Simple ---
    Linear simple_model(28 * 28, num_classes);  // MNIST flatten -> 10 clases
    simple_model.weight.xavier_init();          // Inicialización adecuada
    simple_model.bias.zero();  // Critical!

    for (int epoch = 0; epoch < 5; ++epoch) {
        float loss = 0.0f;
        int correct = 0;
        
        for (int i = 0; i < (int)train_images.size(); ++i)
        {  // Solo 10 iteraciones (1 por muestra)
            simple_model.zero_grad();
            
            // Forward
            Tensor flattened = train_images[i].flatten();  // 28x28 -> 784
            Tensor logits = simple_model.forward(flattened);
            
            // Loss y accuracy
            loss += -log(Activation::softmax(logits)(0, train_labels[i]));
            if (logits.argmax() == train_labels[i]) correct++;
            
            // Backward
            Tensor grad = Activation::softmax_grad(logits, train_labels[i]);
            simple_model.backward(grad);
            
            // Update (LR alto para debug)
            simple_model.update(0.01f);  // LR grande para ver cambios rápidos
            
            // Imprime logits para una muestra (debug)
            if (i == 0)
            {
                std::cout << "Epoch " << epoch << " - Muestra 0 Logits: ";
                for (int j = 0; j < num_classes; ++j) std::cout << logits(0, j) << " ";
                std::cout << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch << " - Loss: " << loss/train_images.size()
                << " | Accuracy: " << (float)correct/train_images.size() * 100 << "%" << std::endl;
    }

    // Evaluación en test cada época (opcional)
    // Evaluar cada época
    int test_correct = 0;
    float test_loss = 0.0f;
    for (size_t i = 0; i < test_images.size(); i++)
    {
        Tensor flattened = test_images[i].flatten();
        Tensor logits = simple_model.forward(flattened);
        test_loss += -log(Activation::softmax(logits)(0, test_labels[i]));
        int predicted = logits.argmax();

        if (predicted == test_labels[i]) test_correct++;

        if (i < 15) // Show a few more examples
        {
            std::cout << "Muestra " << i << " - Predicción: " << predicted
                 << " | Real: " << test_labels[i]
                 << (predicted == test_labels[i] ? " ✓" : " ✗") << endl;
        }
    }

    // // --- Final Evaluation ---
    // cout << "\nEvaluación final en conjunto de prueba:" << endl;
    // int test_correct = 0;
    // float test_loss = 0.0f;
    // for (size_t i = 0; i < test_images.size(); i++)
    // {
    //     Tensor logits = vit.forward(test_images[i]);
    //     test_loss += vit.compute_loss(logits, test_labels[i]);
    //     int predicted = vit.predict(test_images[i]);
    //     if (predicted == test_labels[i])
    //         test_correct++;

    //     if (i < 15) // Show a few more examples
    //     {
    //         cout << "Muestra " << i << " - Predicción: " << predicted
    //              << " | Real: " << test_labels[i]
    //              << (predicted == test_labels[i] ? " ✓" : " ✗") << endl;
    //     }
    // }

    cout << "\nResultados finales:" << endl;
    cout << "- Pérdida: " << fixed << setprecision(4) << test_loss / test_images.size()
         << " | Precisión: " << setprecision(2) << (float)test_correct / test_images.size() * 100 << "%" << endl;

    // // --- Save Model ---
    // auto now = std::chrono::system_clock::now();
    // auto time_t_now = std::chrono::system_clock::to_time_t(now);
    // auto tm = *std::localtime(&time_t_now);
    // std::ostringstream filename;
    
    // if (pretrained_model_path.empty())    
    //     filename << "./models/vit_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".bin";
    // else
    //     filename << "./models/vit_continued_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".bin";
    
    // vit.save_model(filename.str());
    // cout << "Modelo guardado como: " << filename.str() << endl;

    return 0;
}
