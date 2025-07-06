#include "../include/model/patch_embedding.hpp"
#include "../include/core/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

// Función para crear una imagen sintética de prueba
Tensor create_test_image(int channels, int height, int width, int pattern_type = 0)
{
    Tensor image({channels, height, width});
    std::vector<float> image_data(channels * height * width);

    std::mt19937 gen(42); // Seed fijo para reproducibilidad
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int idx = c * (height * width) + h * width + w;

                switch (pattern_type)
                {
                case 0: // Patrón de gradiente
                    image_data[idx] = (float)(h + w) / (height + width);
                    break;
                case 1: // Patrón de cuadrícula
                    image_data[idx] = ((h / 4 + w / 4) % 2) * 0.8f + 0.1f;
                    break;
                case 2: // Patrón circular
                {
                    float center_h = height / 2.0f;
                    float center_w = width / 2.0f;
                    float distance = std::sqrt((h - center_h) * (h - center_h) +
                                               (w - center_w) * (w - center_w));
                    image_data[idx] = 0.5f + 0.4f * std::sin(distance * 0.3f);
                    break;
                }
                default: // Aleatorio
                    image_data[idx] = dist(gen);
                    break;
                }
            }
        }
    }

    image.from_vector(image_data);
    return image;
}

// Función para visualizar un parche
void visualize_patch(const std::vector<float> &patch_data, int patch_size, int channels)
{
    std::cout << "Patch visualization (channel 0, " << patch_size << "x" << patch_size << "):" << std::endl;

    for (int h = 0; h < patch_size; ++h)
    {
        for (int w = 0; w < patch_size; ++w)
        {
            // Solo mostrar el primer canal para simplificar
            float pixel = patch_data[h * patch_size + w];

            // Convertir a caracteres para visualización
            char pixel_char = ' ';
            if (pixel > 0.8f)
                pixel_char = '#';
            else if (pixel > 0.6f)
                pixel_char = '@';
            else if (pixel > 0.4f)
                pixel_char = 'o';
            else if (pixel > 0.2f)
                pixel_char = '.';

            std::cout << pixel_char << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    std::cout << "=== Vision Transformer Patch Embedding Example ===" << std::endl;

    // Parámetros típicos para ViT
    int image_size = 32; // Imagen 32x32 (como CIFAR-10)
    int patch_size = 4;  // Parches de 4x4
    int in_channels = 3; // RGB
    int d_model = 192;   // Dimensión de embedding (ViT-Tiny)

    std::cout << "\n=== CONFIGURATION ===" << std::endl;
    std::cout << "Image size: " << image_size << "x" << image_size << std::endl;
    std::cout << "Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "Input channels: " << in_channels << std::endl;
    std::cout << "Embedding dimension: " << d_model << std::endl;

    int patches_per_dim = image_size / patch_size;
    int num_patches = patches_per_dim * patches_per_dim;
    int patch_vector_size = patch_size * patch_size * in_channels;

    std::cout << "Patches per dimension: " << patches_per_dim << std::endl;
    std::cout << "Total patches: " << num_patches << std::endl;
    std::cout << "Patch vector size: " << patch_vector_size << std::endl;

    // Crear PatchEmbedding
    PatchEmbedding patch_embed(image_size, patch_size, in_channels, d_model);

    // Crear una imagen de prueba
    std::cout << "\n=== INPUT IMAGE ===" << std::endl;
    Tensor test_image = create_test_image(in_channels, image_size, image_size, 1); // Patrón de cuadrícula

    std::cout << "Input image shape: [" << in_channels << ", " << image_size << ", " << image_size << "]" << std::endl;

    // Visualizar una porción de la imagen (canal 0, esquina superior izquierda)
    std::cout << "\nInput image preview (channel 0, top-left 8x8):" << std::endl;
    std::vector<float> img_data = test_image.to_vector();
    for (int h = 0; h < 8; ++h)
    {
        for (int w = 0; w < 8; ++w)
        {
            float pixel = img_data[h * image_size + w]; // Canal 0
            char pixel_char = ' ';
            if (pixel > 0.7f)
                pixel_char = '#';
            else if (pixel > 0.5f)
                pixel_char = '@';
            else if (pixel > 0.3f)
                pixel_char = 'o';
            else if (pixel > 0.1f)
                pixel_char = '.';
            std::cout << pixel_char;
        }
        std::cout << std::endl;
    }

    // Aplicar Patch Embedding
    std::cout << "\n=== APPLYING PATCH EMBEDDING ===" << std::endl;
    std::cout << "Processing: Image → Patching → Linear Projection → Add [CLS] → Add Position Embeddings" << std::endl;

    Tensor embeddings = patch_embed.forward(test_image);

    // Mostrar resultados
    std::cout << "\n=== OUTPUT ===" << std::endl;
    const auto &emb_shape = embeddings.get_shape();
    std::cout << "Output embeddings shape: [" << emb_shape[0] << ", " << emb_shape[1] << "]" << std::endl;
    std::cout << "Sequence length: " << emb_shape[0] << " (1 [CLS] token + " << num_patches << " patches)" << std::endl;
    std::cout << "Embedding dimension: " << emb_shape[1] << std::endl;

    // Mostrar muestras de embeddings
    std::cout << "\nEmbedding samples (first 8 dimensions):" << std::endl;
    std::vector<float> emb_data = embeddings.to_vector();

    // [CLS] token
    std::cout << "[CLS]: ";
    for (int j = 0; j < 8; ++j)
    {
        std::cout << std::fixed << std::setprecision(3) << emb_data[j] << " ";
    }
    std::cout << "..." << std::endl;

    // Primeros parches
    for (int i = 1; i <= std::min(5, num_patches); ++i)
    {
        std::cout << "Patch " << (i - 1) << ": ";
        for (int j = 0; j < 8; ++j)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << emb_data[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    // Verificaciones
    bool shape_correct = (emb_shape[0] == num_patches + 1 && emb_shape[1] == d_model);

    // Verificar que no son todos ceros
    bool has_non_zero = false;
    for (int i = 0; i < 50 && !has_non_zero; ++i)
    {
        if (std::abs(emb_data[i]) > 1e-6)
            has_non_zero = true;
    }

    // Verificar diversidad en los embeddings
    float first_patch_norm = 0.0f, second_patch_norm = 0.0f;
    for (int j = 0; j < d_model; ++j)
    {
        first_patch_norm += emb_data[1 * d_model + j] * emb_data[1 * d_model + j];
        if (num_patches > 1)
        {
            second_patch_norm += emb_data[2 * d_model + j] * emb_data[2 * d_model + j];
        }
    }
    first_patch_norm = std::sqrt(first_patch_norm);
    second_patch_norm = std::sqrt(second_patch_norm);

    std::cout << "\n=== VERIFICATION ===" << std::endl;
    std::cout << "Shape correctness: " << (shape_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Non-zero embeddings: " << (has_non_zero ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Expected shape: [" << (num_patches + 1) << ", " << d_model << "]" << std::endl;
    std::cout << "Actual shape: [" << emb_shape[0] << ", " << emb_shape[1] << "]" << std::endl;

    // Estadísticas
    float mean_val = 0.0f;
    for (float val : emb_data)
        mean_val += val;
    mean_val /= emb_data.size();

    std::cout << "\n=== STATISTICS ===" << std::endl;
    std::cout << "Mean embedding value: " << std::fixed << std::setprecision(6) << mean_val << std::endl;
    std::cout << "First patch norm: " << first_patch_norm << std::endl;
    if (num_patches > 1)
    {
        std::cout << "Second patch norm: " << second_patch_norm << std::endl;
    }

    std::cout << "\n✓ Patch embedding ready for Vision Transformer! 🚀" << std::endl;
    std::cout << "Next step: Feed these embeddings to Transformer encoder layers." << std::endl;

    return 0;
}