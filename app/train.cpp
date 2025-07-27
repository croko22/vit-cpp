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

#include "../include/core/random.h"
#include "../include/core/tensor.h"
#include "../include/core/activation.h"
#include "../include/model/linear.h"
#include "../include/model/layernorm.h"
#include "../include/model/mlp.h"
#include "../include/model/encoder.h"
#include "../include/model/vit.h"
#include "../include/core/optimizer.h"
#include "../include/optimizer/adam.h"

using namespace std;

class MetricsCalculator
{
private:
    int num_classes;
    vector<vector<int>> confusion_matrix;
    long long total_samples = 0;
    long long correct_samples = 0;

public:
    MetricsCalculator(int n_classes) : num_classes(n_classes)
    {
        reset();
    }

    void reset()
    {
        confusion_matrix.assign(num_classes, vector<int>(num_classes, 0));
        total_samples = 0;
        correct_samples = 0;
    }

    void update(int predicted, int actual)
    {
        if (predicted == actual)
        {
            correct_samples++;
        }
        total_samples++;
        confusion_matrix[actual][predicted]++;
    }

    float get_accuracy() const
    {
        return total_samples == 0 ? 0.0f : static_cast<float>(correct_samples) / total_samples;
    }

    float get_macro_f1_score() const
    {
        float f1_sum = 0.0f;
        int class_count = 0;

        for (int i = 0; i < num_classes; ++i)
        {
            long long tp = confusion_matrix[i][i];
            long long fp = 0;
            for (int j = 0; j < num_classes; ++j)
            {
                if (i != j)
                    fp += confusion_matrix[j][i];
            }
            long long fn = 0;
            for (int j = 0; j < num_classes; ++j)
            {
                if (i != j)
                    fn += confusion_matrix[i][j];
            }

            float precision = (tp + fp == 0) ? 0 : static_cast<float>(tp) / (tp + fp);
            float recall = (tp + fn == 0) ? 0 : static_cast<float>(tp) / (tp + fn);

            if (precision + recall > 0)
            {
                f1_sum += 2 * (precision * recall) / (precision + recall);
            }
            class_count++;
        }
        return class_count == 0 ? 0.0f : f1_sum / class_count;
    }
};

class CSVLogger
{
private:
    ofstream file;
    string filename;

public:
    CSVLogger(const string &fname) : filename(fname)
    {
        file.open(filename);
        if (!file.is_open())
        {
            cerr << "Error: No se pudo crear el archivo de log " << filename << endl;
        }

        file << "epoch,train_loss,train_accuracy,train_f1_score,val_loss,val_accuracy,val_f1_score,learning_rate,duration_sec" << endl;
    }

    ~CSVLogger()
    {
        if (file.is_open())
        {
            file.close();
        }
    }

    void log_epoch(int epoch, float train_loss, float train_acc, float train_f1,
                   float val_loss, float val_acc, float val_f1, float lr, double duration)
    {
        if (file.is_open())
        {
            file << epoch << ","
                 << train_loss << "," << train_acc << "," << train_f1 << ","
                 << val_loss << "," << val_acc << "," << val_f1 << ","
                 << lr << "," << duration << endl;
        }
    }
};

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

            Tensor image({28, 28});
            auto &image_data = image.get_data();
            for (int i = 0; i < 784; i++)
            {
                if (!getline(ss, cell, ','))
                    break;
                image_data[i] = stof(cell) / 255.0f;
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

float evaluate_model(VisionTransformer &vit, const vector<Tensor> &images, const vector<int> &labels, MetricsCalculator &metrics)
{
    metrics.reset();
    float total_loss = 0.0f;

    if (images.empty())
        return 0.0f;

    for (size_t i = 0; i < images.size(); i++)
    {
        Tensor logits = vit.forward(images[i]);
        total_loss += vit.compute_loss(logits, labels[i]);
        int predicted = vit.predictWithLogits(logits);
        metrics.update(predicted, labels[i]);
    }

    return total_loss / images.size();
}

void printProgressBar(int count, int total);

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

    cout << "Vision Transformer con Métricas Avanzadas y Logging" << endl;
    cout << "===================================================" << endl;
    Random::seed(23);

    int image_size = 28;
    int patch_size = 7;
    int d_model = 64;
    int num_layers = 1;
    int num_classes = 10;
    int num_heads = 8;
    int d_ff = 256;
    float initial_learning_rate = 3e-4f;
    int epochs = 50;
    int batch_size = 128;
    float val_split_ratio = 0.1f;

    cout << "Cargando datos..." << endl;
    auto [all_train_images, all_train_labels] = DataLoader::load_data(train_filepath, 5000, num_classes);
    auto [test_images, test_labels] = DataLoader::load_data(test_filepath, 1000, num_classes);

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

    VisionTransformer vit(image_size, patch_size, d_model, num_layers, num_classes, num_heads, d_ff);
    float learning_rate = initial_learning_rate;

    auto params = vit.get_parameters();
    std::unique_ptr<Optimizer> optimizer = std::make_unique<Adam>(params, initial_learning_rate);

    auto time_now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(time_now);
    auto tm = *std::localtime(&time_t_now);
    std::ostringstream log_filename_stream;
    log_filename_stream << "./logs/vit_log_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";
    CSVLogger logger(log_filename_stream.str());
    cout << "✅ Archivo de log creado en: " << log_filename_stream.str() << endl;

    if (!pretrained_model_path.empty())
    {
        cout << "Cargando modelo pre-entrenado de: " << pretrained_model_path << endl;
        vit.load_model(pretrained_model_path);
        learning_rate *= 0.1f;
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
    cout << "- Cabezas de atención: " << vit.num_heads << endl;
    cout << "- Dimensión feed-forward (d_ff): " << d_ff << endl;
    cout << "- Capas Transformer: " << num_layers << endl;
    cout << "- Clases: " << num_classes << endl;
    cout << "- Learning rate: " << learning_rate << endl;
    if (!pretrained_model_path.empty())
    {
        cout << " (Learning rate reducido para fine-tuning)" << endl;
    }
    cout << "- Épocas: " << epochs << endl;
    cout << "- Batch size: " << batch_size << endl;
    cout << "- Muestras de entrenamiento: " << train_images.size() << endl;
    cout << "- Muestras de validación: " << val_images.size() << endl;
    cout << "- Muestras de prueba: " << test_images.size() << endl
         << endl;

    cout << "Entrenando..." << endl;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        auto epoch_start_time = chrono::high_resolution_clock::now();

        float train_loss = 0.0f;
        MetricsCalculator train_metrics(num_classes);
        vector<int> train_indices(train_images.size());
        iota(train_indices.begin(), train_indices.end(), 0);
        shuffle(train_indices.begin(), train_indices.end(), Random::gen);

        int total_batches = ceil((float)train_indices.size() / batch_size);
        cout << "\nEpoch " << epoch + 1 << "/" << epochs << endl;

        for (size_t batch_start = 0; batch_start < train_indices.size(); batch_start += batch_size)
        {

            optimizer->zero_grad();
            size_t batch_end = min(batch_start + batch_size, train_indices.size());

            for (size_t i = batch_start; i < batch_end; ++i)
            {
                int idx = train_indices[i];
                Tensor logits = vit.forward(train_images[idx]);
                vit.backward(train_labels[idx]);

                train_loss += vit.compute_loss(logits, train_labels[idx]);
                int predicted = vit.predictWithLogits(logits);
                train_metrics.update(predicted, train_labels[idx]);
            }

            size_t current_batch_size = batch_end - batch_start;
            if (current_batch_size > 0)
            {

                for (auto &p : params)
                {
                    *p.grad = *p.grad * (1.0f / current_batch_size);
                }

                optimizer->step();
            }

            printProgressBar(batch_start / batch_size + 1, total_batches);
            cout << "\r" << flush;
        }
        cout << endl;

        float avg_train_loss = train_images.empty() ? 0 : train_loss / train_images.size();

        MetricsCalculator val_metrics(num_classes);
        float avg_val_loss = evaluate_model(vit, val_images, val_labels, val_metrics);

        auto epoch_end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;

        float train_acc = train_metrics.get_accuracy();
        float train_f1 = train_metrics.get_macro_f1_score();
        float val_acc = val_metrics.get_accuracy();
        float val_f1 = val_metrics.get_macro_f1_score();

        cout << fixed << setprecision(4);
        cout << "  Entrenamiento - Pérdida: " << avg_train_loss << " | Precisión: " << train_acc * 100 << "% | F1-Score: " << train_f1 << endl;
        cout << "  Validación    - Pérdida: " << avg_val_loss << " | Precisión: " << val_acc * 100 << "% | F1-Score: " << val_f1 << endl;
        cout << "  Duración: " << epoch_duration.count() << " s" << endl;

        logger.log_epoch(epoch + 1, avg_train_loss, train_acc, train_f1, avg_val_loss, val_acc, val_f1, learning_rate, epoch_duration.count());
    }

    cout << "\nEvaluación final en conjunto de prueba:" << endl;
    MetricsCalculator test_metrics(num_classes);
    float avg_test_loss = evaluate_model(vit, test_images, test_labels, test_metrics);
    float test_acc = test_metrics.get_accuracy();
    float test_f1 = test_metrics.get_macro_f1_score();

    cout << fixed << setprecision(4);
    cout << "Resultados finales:" << endl;
    cout << "- Pérdida: " << avg_test_loss << " | Precisión: " << test_acc * 100 << "% | F1-Score: " << test_f1 << endl;

    std::ostringstream model_filename;
    model_filename << "./models/vit_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".bin";
    vit.save_model(model_filename.str());
    cout << "Modelo guardado como: " << model_filename.str() << endl;

    return 0;
}

void printProgressBar(int count, int total)
{
    int barWidth = 50;
    float progress = float(count) / total;
    cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
        cout << (i < pos ? "█" : " ");
    cout << "] " << int(progress * 100.0) << "%";
}
