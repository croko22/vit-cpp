#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../include/core/tensor.h"
#include "../include/model/vit.h"

using namespace std;

class CsvDataLoader
{
public:
    static pair<vector<Tensor>, vector<int>> load(const string &file_path, int num_classes)
    {
        vector<Tensor> images;
        vector<int> labels;
        ifstream file(file_path);
        if (!file.is_open())
        {
            throw runtime_error("Error: No se pudo abrir el archivo de datos " + file_path);
        }

        string line;
        getline(file, line); // Omitir cabecera
        while (getline(file, line))
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
        }
        file.close();
        cout << "Datos cargados: " << images.size() << " muestras de " << file_path << endl;
        return {images, labels};
    }
};

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

    file.close();
    cout << "Matriz de confusion guardada en: " << file_path << endl;
}

string get_base_name(const string &path)
{
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of('.');
    string base_name = path.substr(last_slash + 1, last_dot - last_slash - 1);
    return base_name;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cerr << "Uso: " << argv[0] << " <ruta_modelo.bin> <ruta_test.csv> <nombre_dataset>" << endl;
        return 1;
    }

    string model_path = argv[1];
    string test_csv_path = argv[2];
    string dataset_name = argv[3];

    cout << "Evaluador de Modelo ViT" << endl;
    cout << "=======================" << endl;

    cout << "Cargando modelo desde: " << model_path << endl;
    VisionTransformer vit(28, 7, 64, 1, 10, 8, 256);
    try
    {
        vit.load_model(model_path);
        cout << "Modelo cargado exitosamente." << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error fatal al cargar el modelo: " << e.what() << endl;
        return 1;
    }

    cout << "Cargando dataset de prueba desde: " << test_csv_path << endl;
    auto [test_images, test_labels] = CsvDataLoader::load(test_csv_path, vit.num_classes);

    cout << "Calculando predicciones..." << endl;
    vector<vector<int>> confusion_matrix(vit.num_classes, vector<int>(vit.num_classes, 0));
    long long correct_predictions = 0;

    for (size_t i = 0; i < test_images.size(); ++i)
    {
        int true_label = test_labels[i];
        int predicted_label = vit.predict(test_images[i]);

        if (predicted_label >= 0 && predicted_label < vit.num_classes &&
            true_label >= 0 && true_label < vit.num_classes)
        {
            confusion_matrix[true_label][predicted_label]++;
            if (true_label == predicted_label)
            {
                correct_predictions++;
            }
        }
    }
    cout << "Calculo completado." << endl;

    string model_base_name = get_base_name(model_path);
    string output_filename = "./models/confusion_matrix_" + model_base_name + "_" + dataset_name + ".csv";
    save_matrix_to_csv(confusion_matrix, output_filename);

    float accuracy = static_cast<float>(correct_predictions) / test_images.size();
    cout << "\n--- Resumen de Evaluacion ---" << endl;
    cout << "  Muestras totales: " << test_images.size() << endl;
    cout << "  Predicciones correctas: " << correct_predictions << endl;
    cout << "  Precision (Accuracy): " << fixed << setprecision(2) << accuracy * 100.0f << "%" << endl;
    cout << "-----------------------------" << endl;

    return 0;
}
