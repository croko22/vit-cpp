#include "../../include/model/vit.hpp"
#include "../../include/core/ops.hpp" // Para matmul y otros

VisionTransformer::VisionTransformer(int image_size, int patch_size, int in_channels, int num_classes,
                                     int d_model, int num_heads, int d_ff, int num_layers)
    : patch_embedding_(image_size, patch_size, in_channels, d_model),
      final_norm_(d_model),
      classification_head_w_({d_model, num_classes}),
      classification_head_b_({1, num_classes})
{
    for (int i = 0; i < num_layers; ++i)
    {
        encoder_layers_.emplace_back(d_model, num_heads, d_ff);
    }
    // Inicializar gradientes
    classification_head_w_.init_grad();
    classification_head_b_.init_grad();
}

Tensor VisionTransformer::forward(const Tensor &image)
{
    // 1. Convertir imagen a secuencia de embeddings de parches
    Tensor sequence = patch_embedding_.forward(image);

    // 2. Pasar la secuencia a través de la pila de Encoders
    for (auto &layer : encoder_layers_)
    {
        sequence = layer.forward(sequence);
    }

    // 3. Extraer la salida del token [CLS] (siempre es el primero de la secuencia)
    Tensor cls_token_output({1, sequence.get_shape()[1]});
    // Lógica para copiar la primera fila de 'sequence' a 'cls_token_output'...
    // (Esto es una simplificación, tu clase Tensor necesitaría un método para "slice")
    cls_token_output_cache_ = cls_token_output; // Cache para backward

    // 4. Normalización y cabeza de clasificación
    Tensor normalized_output = final_norm_.forward(cls_token_output);
    Tensor logits = matmul(normalized_output, classification_head_w_) + classification_head_b_;

    return logits;
}

// En src/model/vit.cpp
void VisionTransformer::backward(const Tensor &grad_loss)
{
    // En orden inverso al forward

    // 4. Backward a través de la cabeza de clasificación
    auto [grad_norm_output, grad_w] = matmul_backward(grad_loss, normalized_output_cache_, classification_head_w_);
    *(classification_head_w_.grad_) = *(classification_head_w_.grad_) + grad_w;
    *(classification_head_b_.grad_) = *(classification_head_b_.grad_) + sum(grad_loss, 0, true);

    // Backward a través de la normalización final
    Tensor grad_cls_token = final_norm_.backward(grad_norm_output);

    // 3. Crear el gradiente de entrada para la pila de encoders
    // El gradiente solo existe para el token [CLS], el resto es cero.
    Tensor grad_sequence_full(encoder_output_shape_cache_);
    grad_sequence_full.zero_data(); // Pone todo el tensor a cero

    // --- CORRECCIÓN CLAVE AQUÍ ---
    // Colocamos el gradiente del token CLS en la primera fila del gradiente completo.
    grad_sequence_full.set_row(0, grad_cls_token);
    // ----------------------------

    // 2. Backward a través de la pila de Encoders
    for (int i = encoder_layers_.size() - 1; i >= 0; --i)
    {
        grad_sequence_full = encoder_layers_[i].backward(grad_sequence_full);
    }

    // 1. Backward a través de la capa de PatchEmbedding
    patch_embedding_.backward(grad_sequence_full);
}

// void VisionTransformer::backward(const Tensor &grad_loss)
// {
//     // 4. Backward a través de la cabeza de clasificación
//     // Usamos la variable cacheada en lugar del comentario (solución al error 1)
//     auto [grad_norm_output, grad_w] = matmul_backward(grad_loss, normalized_output_cache_, classification_head_w_);
//     *(classification_head_w_.grad_) = grad_w;
//     *(classification_head_b_.grad_) = grad_loss;

//     Tensor grad_cls_token = final_norm_.backward(grad_norm_output);

//     // 3. Crear el gradiente de entrada para la pila de encoders
//     // Usamos la forma cacheada en lugar de un método inexistente (solución al error 2)
//     Tensor grad_sequence_full(encoder_output_shape_cache_);
//     grad_sequence_full.zero_grad();
//     // Aquí necesitarás lógica para colocar `grad_cls_token` en la primera fila de `grad_sequence_full`

//     // 2. Backward a través de la pila de Encoders
//     for (int i = encoder_layers_.size() - 1; i >= 0; --i)
//     {
//         grad_sequence_full = encoder_layers_[i].backward(grad_sequence_full);
//     }

//     // 1. Backward a través de la capa de PatchEmbedding
//     patch_embedding_.backward(grad_sequence_full);
// }

void VisionTransformer::get_parameters(std::vector<Tensor *> &params)
{
    // Limpiar el vector de parámetros
    params.clear();

    // 1. Parámetros del Patch Embedding
    patch_embedding_.get_parameters(params);

    // 2. Parámetros de todas las capas del Encoder
    for (auto &layer : encoder_layers_)
    {
        layer.get_parameters(params);
    }

    // 3. Parámetros de la Layer Normalization final
    final_norm_.get_parameters(params);

    // 4. Parámetros del Classification Head
    params.push_back(&classification_head_w_);
    params.push_back(&classification_head_b_);
}

void VisionTransformer::zero_grad()
{
    std::vector<Tensor *> all_params;
    get_parameters(all_params);

    for (Tensor *param : all_params)
    {
        if (param->grad_)
        {
            param->grad_->zero_grad();
        }
    }
}

void VisionTransformer::get_classification_head_parameters(std::vector<Tensor *> &params)
{
    params.clear();
    params.push_back(&classification_head_w_);
    params.push_back(&classification_head_b_);
}