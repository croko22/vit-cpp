#include "../include/model/multi_head_attention.hpp"
#include "../include/core/tensor.hpp"
#include <iostream>
#include <vector>

int main()
{
    std::cout << "=== MultiHead Attention Example ===" << std::endl;

    // Parámetros del modelo
    int d_model = 512; // Dimensión del modelo (típico en Transformers)
    int num_heads = 8; // Número de cabezales de atención
    int seq_len = 4;   // Longitud de secuencia

    std::cout << "d_model: " << d_model << ", num_heads: " << num_heads
              << ", seq_len: " << seq_len << std::endl;

    // Crear la capa de Multi-Head Attention
    MultiHeadAttention mha(d_model, num_heads);

    // Crear tensor de entrada (seq_len x d_model)
    std::vector<int> input_shape = {seq_len, d_model};
    Tensor input(input_shape);

    // Llenar con datos de ejemplo (simulando embeddings de palabras)
    std::vector<float> input_data(seq_len * d_model);
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {
            // Patrón simple para visualizar mejor
            input_data[i * d_model + j] = 0.1f * (i + 1) + 0.01f * j;
        }
    }
    input.from_vector(input_data);

    std::cout << "\nInput tensor shape: [" << seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Input sample (first 8 values of each sequence):" << std::endl;

    std::vector<float> input_vec = input.to_vector();
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Seq " << i << ": ";
        for (int j = 0; j < 8; ++j)
        { // Solo mostrar primeros 8 valores
            std::cout << input_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Aplicar Multi-Head Attention
    std::cout << "\nApplying Multi-Head Attention..." << std::endl;
    Tensor output = mha.forward(input, input, input);

    // Mostrar resultados
    std::cout << "\nOutput tensor shape: [" << output.get_shape()[0]
              << ", " << output.get_shape()[1] << "]" << std::endl;

    std::cout << "Output sample (first 8 values of each sequence):" << std::endl;
    std::vector<float> output_vec = output.to_vector();
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Seq " << i << ": ";
        for (int j = 0; j < 8; ++j)
        { // Solo mostrar primeros 8 valores
            std::cout << output_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Verificar que las dimensiones se mantienen
    const auto &output_shape = output.get_shape();
    bool shape_correct = (output_shape[0] == seq_len && output_shape[1] == d_model);

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Shape preservation: " << (shape_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Expected: [" << seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Got: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;

    return 0;
}