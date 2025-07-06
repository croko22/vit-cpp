#include "../../include/model/patch_embedding.hpp"
#include "../../include/core/ops.hpp"
#include <stdexcept>
#include <iostream>
#include <random>
#include <cmath>

PatchEmbedding::PatchEmbedding(int image_size, int patch_size, int in_channels, int d_model)
    : patch_size_(patch_size), d_model_(d_model)
{
    if (image_size % patch_size != 0)
    {
        throw std::invalid_argument("El tamaño de la imagen debe ser divisible por el tamaño del parche.");
    }

    int patches_per_dim = image_size / patch_size;
    num_patches_ = patches_per_dim * patches_per_dim;
    int patch_vector_size = patch_size * patch_size * in_channels;

    // --- Inicialización de Parámetros Entrenables ---

    // 1. Proyección lineal para los parches. Shape: [patch_vector_size, d_model]
    projection = Tensor({patch_vector_size, d_model});

    // INICIALIZAR LA PROYECCIÓN CON VALORES ALEATORIOS
    std::mt19937 gen(std::random_device{}());
    float xavier_std = std::sqrt(2.0f / (patch_vector_size + d_model)); // Xavier initialization
    std::normal_distribution<float> dist(0.0f, xavier_std);

    std::vector<float> proj_data(patch_vector_size * d_model);
    for (auto &val : proj_data)
    {
        val = dist(gen);
    }
    projection.from_vector(proj_data);

    // 2. Token [CLS] especial. Es un vector aprendible. Shape: [1, d_model]
    cls_token = Tensor({1, d_model});

    // INICIALIZAR CLS TOKEN
    std::vector<float> cls_data(d_model);
    std::normal_distribution<float> cls_dist(0.0f, 0.02f); // Inicialización pequeña
    for (auto &val : cls_data)
    {
        val = cls_dist(gen);
    }
    cls_token.from_vector(cls_data);

    // 3. Embeddings de posición. Uno para el token [CLS] y uno para cada parche.
    // Shape: [num_patches + 1, d_model]
    position_embeddings = Tensor({num_patches_ + 1, d_model});

    // INICIALIZAR POSITION EMBEDDINGS
    std::vector<float> pos_data((num_patches_ + 1) * d_model);
    std::normal_distribution<float> pos_dist(0.0f, 0.02f); // Inicialización pequeña
    for (auto &val : pos_data)
    {
        val = pos_dist(gen);
    }
    position_embeddings.from_vector(pos_data);

    // Inicializar los gradientes para todos los parámetros
    projection.init_grad();
    cls_token.init_grad();
    position_embeddings.init_grad();

    std::cout << "PatchEmbedding initialized:" << std::endl;
    std::cout << "  - Projection: [" << patch_vector_size << ", " << d_model << "]" << std::endl;
    std::cout << "  - CLS token: [1, " << d_model << "]" << std::endl;
    std::cout << "  - Position embeddings: [" << (num_patches_ + 1) << ", " << d_model << "]" << std::endl;
}

Tensor PatchEmbedding::forward(const Tensor &image)
{
    // Shape de entrada: [Channels, Height, Width]
    const auto &image_shape = image.get_shape();
    int in_channels = image_shape[0];
    int image_height = image_shape[1];
    int image_width = image_shape[2];

    // --- 1. Cortar la imagen en parches y aplanarlos ---
    // Tensor para almacenar todos los parches aplanados
    Tensor flattened_patches({num_patches_, patch_size_ * patch_size_ * in_channels});
    float *patches_data = flattened_patches.get_data();
    const float *image_data = image.get_data();

    int current_patch_idx = 0;
    for (int y = 0; y < image_height; y += patch_size_)
    {
        for (int x = 0; x < image_width; x += patch_size_)
        {
            // Puntero al inicio del vector para el parche actual
            float *current_patch_vector = patches_data + current_patch_idx * flattened_patches.get_shape()[1];
            int vector_idx = 0;

            // Copiar los datos del parche en el vector aplanado
            for (int c = 0; c < in_channels; ++c)
            {
                for (int py = 0; py < patch_size_; ++py)
                {
                    for (int px = 0; px < patch_size_; ++px)
                    {
                        int image_idx = c * (image_height * image_width) + (y + py) * image_width + (x + px);
                        current_patch_vector[vector_idx++] = image_data[image_idx];
                    }
                }
            }
            current_patch_idx++;
        }
    }

    // --- 2. Proyección Lineal de los Parches ---
    // Shape: [num_patches, d_model]
    Tensor projected_patches = matmul(flattened_patches, projection);

    // --- 3. Añadir el token [CLS] al inicio de la secuencia ---
    Tensor sequence_with_cls({num_patches_ + 1, d_model_});
    float *seq_data = sequence_with_cls.get_data();
    const float *cls_data = cls_token.get_data();
    const float *projected_data = projected_patches.get_data();

    // Copiar el token CLS a la primera fila
    std::copy(cls_data, cls_data + d_model_, seq_data);
    // Copiar los parches proyectados a las filas restantes
    std::copy(projected_data, projected_data + num_patches_ * d_model_, seq_data + d_model_);

    // --- 4. Añadir los Embeddings de Posición ---
    Tensor output = sequence_with_cls + position_embeddings;

    // (Opcional, pero recomendado) Cachear tensores necesarios para el backward pass
    // this->flattened_patches_cache_ = flattened_patches;
    // this->sequence_with_cls_cache_ = sequence_with_cls;

    return output;
}

void PatchEmbedding::backward(const Tensor &grad_output)
{
    // grad_output tiene shape [num_patches + 1, d_model]

    // --- 1. Backward a través de la suma de embeddings de posición ---
    // El gradiente fluye directamente a ambos sumandos.
    // Gradiente para los embeddings de posición:
    *(position_embeddings.grad_) = *(position_embeddings.grad_) + grad_output;

    // Gradiente para la secuencia con el token CLS (antes de sumar la posición):
    Tensor grad_sequence_with_cls = grad_output;

    // --- 2. Backward a través de la concatenación del token [CLS] ---
    // Dividimos el gradiente: una parte para el token CLS y otra para los parches.

    // Gradiente para el token CLS (la primera fila de grad_sequence_with_cls)
    Tensor grad_cls_token({1, d_model_});
    std::copy(grad_sequence_with_cls.get_data(), grad_sequence_with_cls.get_data() + d_model_, grad_cls_token.get_data());
    *(cls_token.grad_) = *(cls_token.grad_) + grad_cls_token;

    // Gradiente para los parches proyectados (el resto de las filas)
    Tensor grad_projected_patches({num_patches_, d_model_});
    std::copy(grad_sequence_with_cls.get_data() + d_model_, grad_sequence_with_cls.get_data() + (num_patches_ + 1) * d_model_, grad_projected_patches.get_data());

    // --- 3. Backward a través de la Proyección Lineal ---
    // Necesitamos los parches aplanados originales (que deberíamos haber cacheado en el forward pass).
    // Por ahora, asumimos que tenemos una variable `flattened_patches_cache_`
    // auto [grad_flattened_patches, grad_projection] = matmul_backward(grad_projected_patches, flattened_patches_cache_, projection);
    // *(projection.grad_) = *(projection.grad_) + grad_projection;

    // Nota: El gradiente `grad_flattened_patches` podría usarse para calcular el gradiente
    // con respecto a la imagen de entrada, pero es una operación compleja (un-patching)
    // y a menudo no es necesaria a menos que quieras visualizar los gradientes en la imagen.
    // Para el entrenamiento de los pesos del modelo, los cálculos anteriores son suficientes.
}

void PatchEmbedding::get_parameters(std::vector<Tensor *> &params)
{
    params.push_back(&projection);
    params.push_back(&cls_token);
    params.push_back(&position_embeddings);
}