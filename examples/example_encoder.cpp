#include "../include/model/encoder.hpp"
#include "../include/core/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main()
{
    std::cout << "=== Transformer Encoder Layer Example ===" << std::endl;

    // Parámetros típicos de Transformer
    int d_model = 512; // Dimensión del modelo
    int num_heads = 8; // Número de cabezales de atención
    int d_ff = 2048;   // Dimensión interna FFN (4x d_model)
    int seq_len = 6;   // Longitud de secuencia (como una frase)

    std::cout << "Architecture: d_model=" << d_model
              << ", num_heads=" << num_heads
              << ", d_ff=" << d_ff
              << ", seq_len=" << seq_len << std::endl;

    // Crear una capa del Encoder
    EncoderLayer encoder_layer(d_model, num_heads, d_ff);

    // Crear tensor de entrada simulando embeddings de palabras
    std::vector<int> input_shape = {seq_len, d_model};
    Tensor input(input_shape);

    // Llenar con datos que simulan embeddings reales
    std::vector<float> input_data(seq_len * d_model);
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {
            // Simulamos embeddings con patrones diferentes por palabra
            float word_pattern = 0.3f * std::sin(0.05f * j + i * 0.5f);
            float positional = 0.1f * std::cos(0.02f * j * (i + 1));
            input_data[i * d_model + j] = word_pattern + positional + 0.05f * (i + 1);
        }
    }
    input.from_vector(input_data);

    std::cout << "\n=== INPUT ===" << std::endl;
    std::cout << "Input tensor shape: [" << seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Input sample (first 12 values of each token):" << std::endl;

    std::vector<float> input_vec = input.to_vector();
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < 12; ++j)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << input_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Aplicar Encoder Layer
    std::cout << "\n=== PROCESSING ===" << std::endl;
    std::cout << "Applying Encoder Layer..." << std::endl;
    std::cout << "Process: Input → Multi-Head Attention → Add&Norm → FFN → Add&Norm → Output" << std::endl;

    Tensor output = encoder_layer.forward(input);

    // Mostrar resultados
    std::cout << "\n=== OUTPUT ===" << std::endl;
    std::cout << "Output tensor shape: [" << output.get_shape()[0]
              << ", " << output.get_shape()[1] << "]" << std::endl;

    std::cout << "Output sample (first 12 values of each token):" << std::endl;
    std::vector<float> output_vec = output.to_vector();
    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Token " << i << ": ";
        for (int j = 0; j < 12; ++j)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << output_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Verificaciones
    const auto &output_shape = output.get_shape();
    bool shape_correct = (output_shape[0] == seq_len && output_shape[1] == d_model);

    // Verificar que no son todos ceros
    bool has_non_zero = false;
    for (int i = 0; i < 50 && !has_non_zero; ++i)
    {
        if (std::abs(output_vec[i]) > 1e-6)
            has_non_zero = true;
    }

    // Verificar que la salida es diferente de la entrada
    bool is_different = false;
    for (int i = 0; i < 50 && !is_different; ++i)
    {
        if (std::abs(output_vec[i] - input_vec[i]) > 1e-4)
            is_different = true;
    }

    std::cout << "\n=== VERIFICATION ===" << std::endl;
    std::cout << "Shape preservation: " << (shape_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Non-zero outputs: " << (has_non_zero ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Output differs from input: " << (is_different ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Expected shape: [" << seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Actual shape: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;

    // Estadísticas comparativas
    float input_mean = 0.0f, output_mean = 0.0f;
    float input_var = 0.0f, output_var = 0.0f;

    for (float val : input_vec)
        input_mean += val;
    for (float val : output_vec)
        output_mean += val;
    input_mean /= input_vec.size();
    output_mean /= output_vec.size();

    for (float val : input_vec)
        input_var += (val - input_mean) * (val - input_mean);
    for (float val : output_vec)
        output_var += (val - output_mean) * (val - output_mean);
    input_var /= input_vec.size();
    output_var /= output_vec.size();

    std::cout << "\n=== STATISTICS ===" << std::endl;
    std::cout << "Input  - Mean: " << std::fixed << std::setprecision(6) << input_mean
              << ", Std: " << std::sqrt(input_var) << std::endl;
    std::cout << "Output - Mean: " << output_mean
              << ", Std: " << std::sqrt(output_var) << std::endl;

    // Verificar Layer Normalization (mean ≈ 0, std ≈ 1)
    std::cout << "\nLayer Normalization check (output should have normalized statistics)" << std::endl;

    return 0;
}