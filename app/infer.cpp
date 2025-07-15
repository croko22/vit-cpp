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

int infer(const std::string &model_path, const Tensor &image)
{

    if (image.rows != 28 || image.cols != 28)
    {
        std::cerr << "Error: La imagen de entrada debe ser de 28x28. Dimensiones recibidas: "
                  << image.rows << "x" << image.cols << std::endl;
        return -1;
    }

    int image_size = 28;
    int patch_size = 4;
    int d_model = 64;
    int num_layers = 2;
    int num_classes = 10;

    VisionTransformer vit(image_size, patch_size, d_model, num_layers, num_classes);

    try
    {
        vit.load_model(model_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error al cargar el modelo desde " << model_path << ": " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Error desconocido al cargar el modelo desde " << model_path << std::endl;
        return -1;
    }

    int predicted_class = vit.predict(image);

    return predicted_class;
}

int main(int argc, char *argv[])
{
    Random::seed(chrono::system_clock::now().time_since_epoch().count());

    if (argc < 2 || argc > 3)
    {
        cerr << "Uso: " << argv[0] << " <ruta_modelo> [ruta_imagen_csv]" << endl;
        cerr << "  <ruta_modelo>: Ruta al archivo binario del modelo Vision Transformer entrenado." << endl;
        cerr << "  [ruta_imagen_csv]: Opcional. Ruta a un archivo CSV con imágenes (como fashion-mnist_train.csv)." << endl;
        cerr << "                     Si se omite, se usa una imagen de prueba generada." << endl;
        return 1;
    }

    string model_path = argv[1];

    Tensor test_image(28, 28);
    int true_label = -1;

    if (argc == 3)
    {

        string image_csv_path = argv[2];
        DataGenerator data_gen;

        data_gen.load_mnist_data(image_csv_path, 1, 10, 0.0f, 0.0f);
        auto [train_imgs, train_lbls] = data_gen.get_train_data();
        if (!train_imgs.empty())
        {
            test_image = train_imgs[0];
            true_label = train_lbls[0];
            cout << "Imagen cargada desde: " << image_csv_path << ". Etiqueta real: " << true_label << endl;
        }
        else
        {
            cerr << "Advertencia: No se pudo cargar la imagen desde " << image_csv_path << ". Usando una imagen generada." << endl;

            for (int r = 8; r < 20; ++r)
            {
                for (int c = 8; c < 20; ++c)
                {
                    test_image(r, c) = 1.0f;
                }
            }
        }
    }
    else
    {

        cout << "No se proporcionó ruta de imagen CSV. Generando una imagen de prueba simple (cuadrado blanco)." << endl;
        for (int r = 8; r < 20; ++r)
        {
            for (int c = 8; c < 20; ++c)
            {
                test_image(r, c) = 1.0f;
            }
        }
    }

    cout << "\nRealizando inferencia con el modelo: " << model_path << endl;
    int prediction = infer(model_path, test_image);

    if (prediction != -1)
    {
        cout << "---------------------------------" << endl;
        cout << "      PREDICCIÓN COMPLETADA      " << endl;
        cout << "---------------------------------" << endl;
        cout << "Clase predicha: " << prediction << " 🎉" << endl;
        if (true_label != -1)
        {
            cout << "Etiqueta real (si cargada): " << true_label << endl;
            if (prediction == true_label)
            {
                cout << "¡Coincide con la etiqueta real! ✅" << endl;
            }
            else
            {
                cout << "No coincide con la etiqueta real. ❌" << endl;
            }
        }
        cout << "---------------------------------" << endl;
    }
    else
    {
        cout << "---------------------------------" << endl;
        cout << "  ERROR DURANTE LA INFERENCIA    " << endl;
        cout << "---------------------------------" << endl;
    }

    return 0;
}