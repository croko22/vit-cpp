#include "../include/model/vit.hpp"
#include "../include/core/loss.hpp"
#include "../include/core/optimizer.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

struct MnistDataset
{
    std::vector<Tensor> images;
    std::vector<int> labels;
};

MnistDataset load_fake_mnist_data(int num_samples, int image_size)
{
    MnistDataset db;
    for (int i = 0; i < num_samples; ++i)
    {

        db.images.emplace_back(Tensor({1, image_size, image_size}));

        db.labels.push_back(i % 10);
    }
    return db;
}

int main()
{

    std::cout << "Configurando el entrenamiento del ViT en MNIST..." << std::endl;
    const int IMAGE_SIZE = 28;
    const int PATCH_SIZE = 7;
    const int IN_CHANNELS = 1;
    const int NUM_CLASSES = 10;
    const int D_MODEL = 64;
    const int NUM_HEADS = 4;
    const int D_FF = 128;
    const int NUM_LAYERS = 1;
    const int NUM_EPOCHS = 5;
    const float LEARNING_RATE = 0.01f;

    VisionTransformer model(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, NUM_CLASSES, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS);

    std::vector<Tensor *> params;
    model.get_parameters(params);

    // int count = 0;
    // for (auto &param : params)
    // {

    //     param->print("Parámetro " + std::to_string(count));
    //     count++;
    // }
    // return 0;

    SGD optimizer(params, LEARNING_RATE);
    CrossEntropyLoss loss_fn;

    std::cout << "Cargando datos de MNIST (simulados)..." << std::endl;
    MnistDataset train_data = load_fake_mnist_data(100, IMAGE_SIZE);

    std::cout << "\n--- Iniciando Entrenamiento ---" << std::endl;

    std::cout << "\n--- Iniciando Entrenamiento ---" << std::endl;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        float total_loss = 0.0f;
        std::cout << "\n===============================" << std::endl;
        std::cout << "        INICIANDO EPOCH " << epoch + 1 << std::endl;
        std::cout << "===============================" << std::endl;

        for (size_t i = 0; i < train_data.images.size(); ++i)
        {
            std::cout << "\n--- PASO DE ENTRENAMIENTO " << i << " ---" << std::endl;
            try
            {

                std::cout << "[DEBUG] a. Poniendo gradientes a cero...";
                optimizer.zero_grad();
                std::cout << " [HECHO]" << std::endl;

                std::cout << "[DEBUG] b. Iniciando Forward Pass...";
                Tensor logits = model.forward(train_data.images[i]);
                std::cout << " [HECHO]" << std::endl;

                std::cout << "[DEBUG] c. Calculando la pérdida...";
                float loss = loss_fn.forward(logits, train_data.labels[i]);
                total_loss += loss;
                std::cout << " [HECHO] (Loss: " << loss << ")" << std::endl;

                std::cout << "[DEBUG] d. Iniciando Backward Pass..." << std::endl;

                std::cout << "    [Sub-paso] Calculando gradiente inicial de la pérdida...";
                Tensor initial_grad = loss_fn.backward(logits, train_data.labels[i]); // ← logits shape: [1, num_classes]

                initial_grad.print("Gradiente inicial de la pérdida");

                std::cout << " [HECHO] (Gradiente inicial: " << initial_grad.get_shape()[0] << ")" << std::endl;

                std::cout << " [HECHO]" << std::endl;

                std::cout << "    [Sub-paso] Ejecutando model.backward(initial_grad)..." << std::endl;
                model.backward(initial_grad);
                std::cout << "    [Sub-paso] Backward del modelo completado." << std::endl;

                std::cout << "[DEBUG] d. Backward Pass completado." << std::endl;

                std::cout << "[DEBUG] e. Actualizando pesos (optimizer.step)...";
                optimizer.step();
                std::cout << " [HECHO]" << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                std::cerr << "!!!              EXCEPCIÓN CAPTURADA             !!!" << std::endl;
                std::cerr << "!!! El programa falló aquí. Causa: " << e.what() << std::endl;
                std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                return 1;
            }
        }

        std::cout << "\nEpoch [" << epoch + 1 << "/" << NUM_EPOCHS << "], "
                  << "Loss Promedio: " << total_loss / train_data.images.size() << std::endl;
    }
    std::cout << "\n--- Entrenamiento Finalizado ---" << std::endl;

    return 0;
}