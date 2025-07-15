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
#include "../include/core/random.h"
#include "../include/core/tensor.h"
#include "../include/core/activation.h"
#include "../include/model/linear.h"
#include "../include/model/layernorm.h"
#include "../include/model/mlp.h"
#include "../include/model/encoder.h"
#include "../include/model/vit.h"

using namespace std;

class DataGenerator
{
public:
    std::vector<Tensor> train_images, val_images, test_images;
    std::vector<int> train_labels, val_labels, test_labels;

    void load_mnist_data(const string &filename, int max_samples_to_load = 1000, int num_classes_to_load = 10, float val_split = 0.1f, float test_split = 0.1f)
    {
        vector<Tensor> all_images;
        vector<int> all_labels;

        ifstream file(filename);
        if (!file.is_open())
        {
            cout << "Error: No se pudo abrir el archivo " << filename << endl;
            return;
        }

        string line;
        int samples_loaded = 0;

        getline(file, line);
        while (getline(file, line) && samples_loaded < max_samples_to_load)
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

                float pixel_value = stof(cell) / 255.0f;
                int row = i / 28;
                int col = i % 28;
                image(row, col) = pixel_value;
            }
            all_images.push_back(image);
            all_labels.push_back(label);
            samples_loaded++;
        }
        file.close();

        vector<int> indices(all_images.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), Random::gen);

        int total = all_images.size();
        int val_size = static_cast<int>(total * val_split);
        int test_size = static_cast<int>(total * test_split);
        int train_size = total - val_size - test_size;

        for (int idx : indices)
        {
            if (train_images.size() < train_size)
            {
                train_images.push_back(all_images[idx]);
                train_labels.push_back(all_labels[idx]);
            }
            else if (val_images.size() < val_size)
            {
                val_images.push_back(all_images[idx]);
                val_labels.push_back(all_labels[idx]);
            }
            else
            {
                test_images.push_back(all_images[idx]);
                test_labels.push_back(all_labels[idx]);
            }
        }

        cout << "Datos cargados: " << samples_loaded << " muestras de " << filename << endl;
        cout << "- Entrenamiento: " << train_images.size() << endl;
        cout << "- Validación: " << val_images.size() << endl;
        cout << "- Prueba: " << test_images.size() << endl;
    }

    tuple<vector<Tensor>, vector<int>> get_train_data()
    {
        return make_tuple(train_images, train_labels);
    }

    tuple<vector<Tensor>, vector<int>> get_val_data()
    {
        return make_tuple(val_images, val_labels);
    }

    tuple<vector<Tensor>, vector<int>> get_test_data()
    {
        return make_tuple(test_images, test_labels);
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

int main()
{
    cout << "Vision Transformer con Entrenamiento por Batch" << endl;
    cout << "==============================================" << endl;
    Random::seed(42);

    int image_size = 28;
    int patch_size = 4;
    int d_model = 64;
    int num_layers = 2;
    int num_classes = 10;
    float learning_rate = 3e-4f;
    int epochs = 10;
    int batch_size = 128;

    VisionTransformer vit(image_size, patch_size, d_model, num_layers, num_classes);

    DataGenerator data_gen;

    data_gen.load_mnist_data("./data/mnist/fashion-mnist_train.csv", 10000, num_classes);

    auto [train_images, train_labels] = data_gen.get_train_data();
    auto [val_images, val_labels] = data_gen.get_val_data();
    auto [test_images, test_labels] = data_gen.get_test_data();

    cout << "\nConfiguración:" << endl;
    cout << "- Imagen: " << image_size << "x" << image_size << endl;
    cout << "- Patch: " << patch_size << "x" << patch_size << endl;
    cout << "- Patches por imagen: " << vit.num_patches << endl;
    cout << "- Dimensión de embedding (d_model): " << d_model << endl;
    cout << "- Capas Transformer: " << num_layers << endl;
    cout << "- Clases: " << num_classes << endl;
    cout << "- Learning rate: " << learning_rate << endl;
    cout << "- Épocas: " << epochs << endl;
    cout << "- Batch size: " << batch_size << endl;
    cout << "- Muestras de entrenamiento: " << train_images.size() << endl;
    cout << "- Muestras de validación: " << val_images.size() << endl;
    cout << "- Muestras de prueba: " << test_images.size() << endl
         << endl;

    cout << "Entrenando..." << endl;

    for (int epoch = 0; epoch < epochs; epoch++)
    {

        float train_loss = 0.0f;
        int train_correct = 0;
        vector<int> indices(train_images.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), Random::gen);

        int batch_count = 0;
        int total_batches = ceil((float)indices.size() / batch_size);

        cout << "Epoch " << epoch + 1 << "/" << epochs << " - Progreso: ";
        for (size_t batch_start = 0; batch_start < indices.size(); batch_start += batch_size)
        {
            size_t batch_end = min(batch_start + batch_size, indices.size());
            vector<int> current_batch_indices;
            for (size_t i = batch_start; i < batch_end; ++i)
            {
                current_batch_indices.push_back(indices[i]);
            }

            vit.zero_grad();

            for (int idx : current_batch_indices)
            {

                Tensor logits = vit.forward(train_images[idx]);

                vit.backward(train_labels[idx]);

                train_loss += vit.compute_loss(logits, train_labels[idx]);
                if (vit.predict(train_images[idx]) == train_labels[idx])
                    train_correct++;
            }

            vit.update_weights(learning_rate);

            batch_count++;
            printProgressBar(batch_count, total_batches);
        }
        cout << endl;

        float val_loss = 0.0f;
        int val_correct = 0;
        for (size_t i = 0; i < val_images.size(); i++)
        {
            Tensor logits = vit.forward(val_images[i]);
            val_loss += vit.compute_loss(logits, val_labels[i]);
            if (vit.predict(val_images[i]) == val_labels[i])
                val_correct++;
        }

        float avg_train_loss = train_loss / train_images.size();
        float train_acc = (float)train_correct / train_images.size();
        float val_acc = (float)val_correct / val_images.size();

        if (epoch % 1 == 0 || epoch == epochs - 1)
        {
            cout << "Época " << epoch + 1 << "/" << epochs << endl;
            cout << "  Entrenamiento - Pérdida: " << fixed << setprecision(4) << avg_train_loss
                 << " | Precisión: " << setprecision(2) << train_acc * 100 << "%" << endl;
            cout << "  Validación  - Pérdida: " << fixed << setprecision(4) << val_loss / val_images.size()
                 << " | Precisión: " << setprecision(2) << val_acc * 100 << "%" << endl
                 << endl;
        }
    }

    cout << "\nEvaluación final en conjunto de prueba:" << endl;
    int test_correct = 0;
    float test_loss = 0.0f;
    for (size_t i = 0; i < test_images.size(); i++)
    {
        Tensor logits = vit.forward(test_images[i]);
        test_loss += vit.compute_loss(logits, test_labels[i]);
        int predicted = vit.predict(test_images[i]);
        if (predicted == test_labels[i])
            test_correct++;

        if (i < 10)
        {
            cout << "Muestra " << i << " - Predicción: " << predicted
                 << " | Real: " << test_labels[i]
                 << (predicted == test_labels[i] ? " ✓" : " ✗") << endl;
        }
    }

    cout << "\nResultados finales:" << endl;
    cout << "- Pérdida: " << fixed << setprecision(4) << test_loss / test_images.size()
         << " | Precisión: " << setprecision(2) << (float)test_correct / test_images.size() * 100 << "%" << endl;

    return 0;
}
