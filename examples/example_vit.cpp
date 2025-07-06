#include "../include/model/vit.hpp"
#include "../include/core/loss.hpp"
#include "../include/core/optimizer.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

// --- NECESITARÁS UN CARGADOR DE DATOS MNIST ---
// Por simplicidad, aquí simularemos los datos.
// En un caso real, aquí iría el código para leer los archivos de MNIST.
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
        // Imagen falsa de 1 canal, 28x28
        db.images.emplace_back(Tensor({1, image_size, image_size}));
        // Etiqueta falsa (0-9)
        db.labels.push_back(i % 10);
    }
    return db;
}

int main()
{
    // --- 1. Hiperparámetros y Configuración ---
    std::cout << "Configurando el entrenamiento del ViT en MNIST..." << std::endl;
    const int IMAGE_SIZE = 28;  // MNIST tiene imágenes de 28x28
    const int PATCH_SIZE = 7;   // 28 es divisible por 7
    const int IN_CHANNELS = 1;  // MNIST es en escala de grises
    const int NUM_CLASSES = 10; // Dígitos del 0 al 9
    const int D_MODEL = 64;     // Dimensión pequeña para una demo
    const int NUM_HEADS = 4;
    const int D_FF = 128;
    const int NUM_LAYERS = 2;
    const int NUM_EPOCHS = 5;
    const float LEARNING_RATE = 0.01f;

    // --- 2. Instanciar Modelo, Pérdida y Optimizador ---
    VisionTransformer model(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, NUM_CLASSES, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS);

    std::vector<Tensor *> params;
    model.get_parameters(params); // Obtener todos los pesos y biases

    SGD optimizer(params, LEARNING_RATE);
    CrossEntropyLoss loss_fn;

    // --- 3. Cargar Datos ---
    std::cout << "Cargando datos de MNIST (simulados)..." << std::endl;
    MnistDataset train_data = load_fake_mnist_data(100, IMAGE_SIZE); // Usamos 100 muestras para la demo

    // --- 4. Bucle de Entrenamiento ---
    std::cout << "\n--- Iniciando Entrenamiento ---" << std::endl;
    // --- 4. Bucle de Entrenamiento (CON DEPURACIÓN) ---
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
                // a. Poner a cero los gradientes
                std::cout << "[DEBUG] a. Poniendo gradientes a cero...";
                optimizer.zero_grad();
                std::cout << " [HECHO]" << std::endl;

                // b. Forward Pass
                std::cout << "[DEBUG] b. Iniciando Forward Pass...";
                Tensor logits = model.forward(train_data.images[i]);
                std::cout << " [HECHO]" << std::endl;

                // c. Calcular la pérdida
                std::cout << "[DEBUG] c. Calculando la pérdida...";
                float loss = loss_fn.forward(logits, train_data.labels[i]);
                total_loss += loss;
                std::cout << " [HECHO] (Loss: " << loss << ")" << std::endl;

                // d. Backward Pass
                std::cout << "[DEBUG] d. Iniciando Backward Pass..." << std::endl;

                std::cout << "    [Sub-paso] Calculando gradiente inicial de la pérdida...";
                Tensor initial_grad = loss_fn.backward();

                std::cout << " [HECHO] (Gradiente inicial: " << initial_grad.get_shape()[0] << ")" << std::endl;

                std::cout << " [HECHO]" << std::endl;

                std::cout << "    [Sub-paso] Ejecutando model.backward(initial_grad)..." << std::endl;
                model.backward(initial_grad);
                std::cout << "    [Sub-paso] Backward del modelo completado." << std::endl;

                std::cout << "[DEBUG] d. Backward Pass completado." << std::endl;

                // e. Actualizar los pesos
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
                return 1; // Terminar el programa
            }
        }

        std::cout << "\nEpoch [" << epoch + 1 << "/" << NUM_EPOCHS << "], "
                  << "Loss Promedio: " << total_loss / train_data.images.size() << std::endl;
    }
    std::cout << "\n--- Entrenamiento Finalizado ---" << std::endl;

    return 0;
}