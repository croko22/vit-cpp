#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <map>
#include <sys/stat.h>

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
#include "../include/optimizer/adamw.h"
#include "../include/optimizer/sgd.h" // Asumiendo que existe

using namespace std;

struct TrainingConfig
{
    string dataset_name;
    string train_filepath;
    string test_filepath;
    string pretrained_model_path = "";
    int image_size = 28;
    int patch_size = 4;
    int d_model = 128;
    int num_layers = 4;
    int num_heads = 8;
    int d_ff = 512;
    int num_classes = 10;
    string optimizer_type = "adamw";
    float learning_rate = 1e-4f;
    int epochs = 100;
    int batch_size = 128;
    float val_split_ratio = 0.1f;
    int max_train_samples = 20000;
    int max_test_samples = 5000;
};

TrainingConfig load_config_from_file(const string &filepath)
{
    TrainingConfig config;
    ifstream file(filepath);
    if (!file.is_open())
    {
        throw runtime_error("Error: No se pudo abrir el archivo de configuracion " + filepath);
    }

    map<string, string> params;
    string line;
    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        stringstream ss(line);
        string key, value;
        if (getline(ss, key, '=') && getline(ss, value))
        {
            key.erase(remove_if(key.begin(), key.end(), ::isspace), key.end());
            value.erase(remove_if(value.begin(), value.end(), ::isspace), value.end());
            params[key] = value;
        }
    }

    try
    {
        config.dataset_name = params.at("dataset_name");
        config.train_filepath = params.at("train_filepath");
        config.test_filepath = params.at("test_filepath");
        if (params.count("pretrained_model_path"))
            config.pretrained_model_path = params["pretrained_model_path"];

        config.image_size = stoi(params.at("image_size"));
        config.patch_size = stoi(params.at("patch_size"));
        config.d_model = stoi(params.at("d_model"));
        config.num_layers = stoi(params.at("num_layers"));
        config.num_heads = stoi(params.at("num_heads"));
        config.d_ff = stoi(params.at("d_ff"));

        if (params.count("optimizer_type"))
            config.optimizer_type = params["optimizer_type"];
        config.learning_rate = stof(params.at("learning_rate"));

        config.epochs = stoi(params.at("epochs"));
        config.batch_size = stoi(params.at("batch_size"));
        config.val_split_ratio = stof(params.at("val_split_ratio"));
        config.max_train_samples = stoi(params.at("max_train_samples"));
        config.max_test_samples = stoi(params.at("max_test_samples"));
    }
    catch (const out_of_range &e)
    {
        throw runtime_error("Error: Parametro de configuracion faltante o invalido.");
    }

    return config;
}

class MetricsCalculator
{
private:
    int num_classes;
    vector<vector<int>> confusion_matrix;
    long long total_samples = 0;
    long long correct_samples = 0;

public:
    MetricsCalculator(int n_classes) : num_classes(n_classes) { reset(); }
    void reset()
    {
        confusion_matrix.assign(num_classes, vector<int>(num_classes, 0));
        total_samples = 0;
        correct_samples = 0;
    }
    void update(int predicted, int actual)
    {
        if (predicted == actual)
            correct_samples++;
        if (actual < num_classes && predicted < num_classes)
            confusion_matrix[actual][predicted]++;
        total_samples++;
    }
    float get_accuracy() const
    {
        return total_samples == 0 ? 0.0f : static_cast<float>(correct_samples) / total_samples;
    }
    const vector<vector<int>> &get_confusion_matrix() const
    {
        return confusion_matrix;
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
                if (i != j)
                    fp += confusion_matrix[j][i];
            long long fn = 0;
            for (int j = 0; j < num_classes; ++j)
                if (i != j)
                    fn += confusion_matrix[i][j];
            float precision = (tp + fp == 0) ? 0 : static_cast<float>(tp) / (tp + fp);
            float recall = (tp + fn == 0) ? 0 : static_cast<float>(tp) / (tp + fn);
            if (precision + recall > 0)
                f1_sum += 2 * (precision * recall) / (precision + recall);
            class_count++;
        }
        return class_count == 0 ? 0.0f : f1_sum / class_count;
    }
};

class CsvLogger
{
private:
    ofstream file;

public:
    CsvLogger(const string &file_path)
    {
        file.open(file_path);
        if (!file.is_open())
            throw runtime_error("Error: No se pudo crear el archivo de log " + file_path);
        file << "epoch,train_loss,train_accuracy,train_f1_score,val_loss,val_accuracy,val_f1_score,learning_rate,duration_sec" << endl;
    }
    void log_epoch(int epoch, float train_loss, float train_acc, float train_f1, float val_loss, float val_acc, float val_f1, float lr, double duration)
    {
        if (file.is_open())
            file << epoch << "," << train_loss << "," << train_acc << "," << train_f1 << "," << val_loss << "," << val_acc << "," << val_f1 << "," << lr << "," << duration << endl;
    }
};

class CsvDataLoader
{
public:
    static pair<vector<Tensor>, vector<int>> load(const string &file_path, int num_classes, int max_samples = -1)
    {
        vector<Tensor> images;
        vector<int> labels;
        ifstream file(file_path);
        if (!file.is_open())
            throw runtime_error("Error: No se pudo abrir el archivo de datos " + file_path);
        string line;
        getline(file, line);
        int samples_loaded = 0;
        while (getline(file, line) && (max_samples == -1 || samples_loaded < max_samples))
        {
            stringstream ss(line);
            string cell;
            if (!getline(ss, cell, ','))
                continue;
            int label = stoi(cell);
            if (label >= num_classes)
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
        cout << "Datos cargados: " << samples_loaded << " muestras de " << file_path << endl;
        return {images, labels};
    }
};

int get_num_classes_for_dataset(const string &dataset_name)
{
    if (dataset_name == "mnist" || dataset_name == "fashionmnist")
        return 10;
    if (dataset_name == "bloodmnist")
        return 8;
    throw invalid_argument("Dataset no reconocido: " + dataset_name);
}

void save_config_to_file(const string &file_path, const TrainingConfig &config, float final_accuracy, float final_f1_score)
{
    ofstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Advertencia: No se pudo guardar el archivo de configuracion " << file_path << endl;
        return;
    }
    file << "dataset_name = " << config.dataset_name << endl;
    file << "image_size = " << config.image_size << endl;
    file << "patch_size = " << config.patch_size << endl;
    file << "d_model = " << config.d_model << endl;
    file << "num_layers = " << config.num_layers << endl;
    file << "num_heads = " << config.num_heads << endl;
    file << "d_ff = " << config.d_ff << endl;
    file << "num_classes = " << config.num_classes << endl;
    file << "optimizer_type = " << config.optimizer_type << endl;
    file << "learning_rate = " << config.learning_rate << endl;
    file << "epochs = " << config.epochs << endl;
    file << "batch_size = " << config.batch_size << endl;
    file << "final_test_accuracy = " << final_accuracy << endl;
    file << "final_test_f1_score = " << final_f1_score << endl;
    cout << "Configuracion de entrenamiento guardada en: " << file_path << endl;
}

void save_matrix_to_csv(const vector<vector<int>> &matrix, const string &file_path)
{
    ofstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Error: No se pudo crear el archivo " << file_path << endl;
        return;
    }
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            file << matrix[i][j] << (j == matrix[i].size() - 1 ? "" : ",");
        }
        file << endl;
    }
    cout << "Matriz de confusion guardada en: " << file_path << endl;
}

float evaluate_model(VisionTransformer &vit, const vector<Tensor> &images, const vector<int> &labels, MetricsCalculator &metrics)
{
    metrics.reset();
    if (images.empty())
        return 0.0f;
    float total_loss = 0.0f;
    for (size_t i = 0; i < images.size(); i++)
    {
        Tensor logits = vit.forward(images[i]);
        total_loss += vit.compute_loss(logits, labels[i]);
        metrics.update(vit.predictWithLogits(logits), labels[i]);
    }
    return total_loss / images.size();
}

void print_progress_bar(int count, int total)
{
    const int bar_width = 50;
    float progress = static_cast<float>(count) / total;
    cout << "[";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i)
        cout << (i < pos ? "█" : " ");
    cout << "] " << static_cast<int>(progress * 100.0) << " %\r" << flush;
}

void create_directories(const string &dataset_name)
{
    string models_path = "./models/" + dataset_name;
    string logs_path = "./logs/" + dataset_name;
    mkdir(models_path.c_str(), 0777);
    mkdir(logs_path.c_str(), 0777);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Uso: " << argv[0] << " <ruta_a_config_file>" << endl;
        return 1;
    }

    TrainingConfig config;
    try
    {
        config = load_config_from_file(argv[1]);
        config.num_classes = get_num_classes_for_dataset(config.dataset_name);
        create_directories(config.dataset_name);
    }
    catch (const exception &e)
    {
        cerr << "Error fatal al cargar la configuracion: " << e.what() << endl;
        return 1;
    }

    cout << "Entrenamiento de Vision Transformer" << endl;
    cout << "===================================" << endl;
    Random::seed(42);

    auto [all_train_images, all_train_labels] = CsvDataLoader::load(config.train_filepath, config.num_classes, config.max_train_samples);
    auto [test_images, test_labels] = CsvDataLoader::load(config.test_filepath, config.num_classes, config.max_test_samples);

    vector<int> indices(all_train_images.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), Random::gen);

    size_t val_size = static_cast<size_t>(all_train_images.size() * config.val_split_ratio);
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

    VisionTransformer vit(config.image_size, config.patch_size, config.d_model, config.num_layers, config.num_classes, config.num_heads, config.d_ff);

    if (!config.pretrained_model_path.empty())
    {
        cout << "Cargando modelo pre-entrenado: " << config.pretrained_model_path << endl;
        vit.load_model(config.pretrained_model_path);
        config.learning_rate *= 0.1f;
    }

    auto params = vit.get_parameters();
    unique_ptr<Optimizer> optimizer;
    if (config.optimizer_type == "adamw")
    {
        optimizer = make_unique<AdamW>(params, config.learning_rate);
    }
    else if (config.optimizer_type == "adam")
    {
        optimizer = make_unique<Adam>(params, config.learning_rate);
    }
    else if (config.optimizer_type == "sgd")
    {
        optimizer = make_unique<SGD>(params, config.learning_rate);
    }
    else
    {
        throw runtime_error("Optimizador no reconocido: " + config.optimizer_type);
    }
    cout << "Optimizador seleccionado: " << config.optimizer_type << endl;

    auto time_now = chrono::system_clock::now();
    auto time_t_now = chrono::system_clock::to_time_t(time_now);
    tm local_tm = *localtime(&time_t_now);
    stringstream time_ss;
    time_ss << put_time(&local_tm, "%Y%m%d_%H%M%S");
    string timestamp = time_ss.str();

    string log_dir = "./logs/" + config.dataset_name + "/";
    string log_filepath = log_dir + "log_" + timestamp + ".csv";
    CsvLogger logger(log_filepath);
    cout << "Archivo de log creado en: " << log_filepath << endl;

    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        auto epoch_start_time = chrono::high_resolution_clock::now();
        float train_loss = 0.0f;
        MetricsCalculator train_metrics(config.num_classes);
        vector<int> train_indices(train_images.size());
        iota(train_indices.begin(), train_indices.end(), 0);
        shuffle(train_indices.begin(), train_indices.end(), Random::gen);

        cout << "\nEpoch " << epoch + 1 << "/" << config.epochs << endl;
        int total_batches = (train_images.size() + config.batch_size - 1) / config.batch_size;

        for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx)
        {
            optimizer->zero_grad();
            size_t batch_start = batch_idx * config.batch_size;
            size_t batch_end = min(batch_start + config.batch_size, train_images.size());
            for (size_t i = batch_start; i < batch_end; ++i)
            {
                int idx = train_indices[i];
                Tensor logits = vit.forward(train_images[idx]);
                vit.backward(train_labels[idx]);
                train_loss += vit.compute_loss(logits, train_labels[idx]);
                train_metrics.update(vit.predictWithLogits(logits), train_labels[idx]);
            }
            size_t current_batch_size = batch_end - batch_start;
            if (current_batch_size > 0)
            {
                for (auto &p : params)
                    *p.grad = *p.grad * (1.0f / current_batch_size);
                optimizer->step();
            }
            print_progress_bar(batch_idx + 1, total_batches);
        }
        cout << endl;

        MetricsCalculator val_metrics(config.num_classes);
        float avg_val_loss = evaluate_model(vit, val_images, val_labels, val_metrics);
        auto epoch_end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;

        cout << fixed << setprecision(4);
        cout << "  Entrenamiento - Perdida: " << train_loss / train_images.size() << " | Acc: " << train_metrics.get_accuracy() * 100 << "% | F1: " << train_metrics.get_macro_f1_score() << endl;
        cout << "  Validacion    - Perdida: " << avg_val_loss << " | Acc: " << val_metrics.get_accuracy() * 100 << "% | F1: " << val_metrics.get_macro_f1_score() << endl;

        logger.log_epoch(epoch + 1, train_loss / train_images.size(), train_metrics.get_accuracy(), train_metrics.get_macro_f1_score(),
                         avg_val_loss, val_metrics.get_accuracy(), val_metrics.get_macro_f1_score(), config.learning_rate, epoch_duration.count());
    }

    MetricsCalculator test_metrics(config.num_classes);
    float avg_test_loss = evaluate_model(vit, test_images, test_labels, test_metrics);
    float test_acc = test_metrics.get_accuracy();
    float test_f1 = test_metrics.get_macro_f1_score();

    cout << "\n--- Evaluacion Final ---" << endl;
    cout << "  Perdida: " << avg_test_loss << " | Acc: " << test_acc * 100 << "% | F1: " << test_f1 << endl;

    stringstream acc_ss;
    acc_ss << fixed << setprecision(2) << test_acc * 100.0f;
    string model_base_name = "vit_" + timestamp + "_acc_" + acc_ss.str();
    string model_dir = "./models/" + config.dataset_name + "/";
    string model_bin_path = model_dir + model_base_name + ".bin";
    string model_config_path = model_dir + model_base_name + ".txt";
    string confusion_matrix_path = model_dir + "cm_" + model_base_name + ".csv";

    vit.save_model(model_bin_path);
    cout << "Modelo guardado en: " << model_bin_path << endl;

    save_config_to_file(model_config_path, config, test_acc, test_f1);
    save_matrix_to_csv(test_metrics.get_confusion_matrix(), confusion_matrix_path);

    return 0;
}
