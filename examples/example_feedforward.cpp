#include "../include/model/feedforward.hpp"
#include "../include/core/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

int main()
{
    std::cout << "=== FeedForward Network Example ===" << std::endl;

    // Parámetros típicos de Transformer
    int d_model = 512; // Dimensión del modelo
    int d_ff = 2048;   // Dimensión interna FFN (típicamente 4x d_model)
    int seq_len = 4;   // Longitud de secuencia

    std::cout << "d_model: " << d_model << ", d_ff: " << d_ff
              << " (expansion factor: " << (float)d_ff / d_model << "x)" << std::endl;

    // Crear la red FeedForward
    FeedForwardNetwork ffn(d_model, d_ff);

    // Crear tensor de entrada (seq_len x d_model)
    std::vector<int> input_shape = {seq_len, d_model};
    Tensor input(input_shape);

    // Llenar con datos de ejemplo (simulando salida de attention)
    std::vector<float> input_data(seq_len * d_model);
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {
            // Patrón que simula activaciones reales
            input_data[i * d_model + j] = 0.5f * std::sin(0.1f * j) + 0.1f * (i + 1);
        }
    }
    input.from_vector(input_data);

    std::cout << "\nInput tensor shape: [" << seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Input sample (first 10 values of each sequence):" << std::endl;

    std::vector<float> input_vec = input.to_vector();
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Seq " << i << ": ";
        for (int j = 0; j < 10; ++j)
        {
            std::cout << std::fixed << std::setprecision(3) << input_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Aplicar FeedForward Network
    std::cout << "\nApplying FeedForward Network..." << std::endl;
    std::cout << "Process: input[" << seq_len << "," << d_model << "] → W1 → ReLU → W2 → output["
              << seq_len << "," << d_model << "]" << std::endl;

    Tensor output = ffn.forward(input);

    // Mostrar resultados
    std::cout << "\nOutput tensor shape: [" << output.get_shape()[0]
              << ", " << output.get_shape()[1] << "]" << std::endl;

    std::cout << "Output sample (first 10 values of each sequence):" << std::endl;
    std::vector<float> output_vec = output.to_vector();
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Seq " << i << ": ";
        for (int j = 0; j < 10; ++j)
        {
            std::cout << std::fixed << std::setprecision(3) << output_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Verificaciones
    const auto &output_shape = output.get_shape();
    bool shape_correct = (output_shape[0] == seq_len && output_shape[1] == d_model);

    // Verificar que no son todos ceros
    bool has_non_zero = false;
    for (int i = 0; i < 20 && !has_non_zero; ++i)
    {
        if (std::abs(output_vec[i]) > 1e-6)
            has_non_zero = true;
    }

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Shape preservation: " << (shape_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Non-zero outputs: " << (has_non_zero ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Expected shape: [" << seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Actual shape: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;

    // Mostrar estadísticas básicas
    float min_val = *std::min_element(output_vec.begin(), output_vec.end());
    float max_val = *std::max_element(output_vec.begin(), output_vec.end());
    float mean_val = 0.0f;
    for (float val : output_vec)
        mean_val += val;
    mean_val /= output_vec.size();

    std::cout << "\nOutput statistics:" << std::endl;
    std::cout << "Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val << std::endl;

    return 0;
}