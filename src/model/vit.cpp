#include "../../include/model/vit.hpp"
#include "../../include/core/ops.hpp"
#include <iostream>

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
        encoder_output_shape_cache_ = sequence.get_shape(); // 👈 Esto FALTA
    }

    // 3. Extraer la salida del token [CLS] (siempre es el primero de la secuencia)
    Tensor cls_token_output({1, sequence.get_shape()[1]});
    cls_token_output_cache_ = cls_token_output;
    // Lógica para copiar la primera fila de 'sequence' a 'cls_token_output'...
    // (Esto es una simplificación, tu clase Tensor necesitaría un método para "slice")

    // 4. Normalización y cabeza de clasificación
    Tensor normalized_output = final_norm_.forward(cls_token_output);
    normalized_output_cache_ = normalized_output; // Guardar para backward
    Tensor logits = matmul(normalized_output, classification_head_w_) + classification_head_b_;

    return logits;
}

// En src/model/vit.cpp
void VisionTransformer::backward(const Tensor &grad_loss)
{
    std::cout << "[ViT::backward] grad_loss shape: ";
    grad_loss.print("grad_loss");

    std::cout << "[ViT::backward] normalized_output_cache_ shape: ";
    normalized_output_cache_.print("normalized_output_cache_");

    std::cout << "[ViT::backward] classification_head_w_ shape: ";
    classification_head_w_.print("classification_head_w_");

    // 4. Backward a través de la cabeza de clasificación
    std::cout << "[DEBUG] Shapes justo antes de matmul_backward()" << std::endl;
    std::cout << "grad_loss: ";
    grad_loss.print("grad_loss");
    std::cout << "normalized_output_cache_: ";
    normalized_output_cache_.print("normalized_output_cache_");
    std::cout << "classification_head_w_: ";
    classification_head_w_.print("classification_head_w_");

    auto [grad_norm_output, grad_w] = matmul_backward(grad_loss, normalized_output_cache_, classification_head_w_);

    std::cout << "[ViT::backward] <- grad_norm_output shape: ";
    grad_norm_output.print("grad_norm_output");

    *(classification_head_w_.grad_) = *(classification_head_w_.grad_) + grad_w;
    *(classification_head_b_.grad_) = *(classification_head_b_.grad_) + sum(grad_loss, 0, true);

    // Backward a través de la normalización final
    std::cout << "[ViT::backward] -> final_norm.backward..." << std::endl;
    Tensor grad_cls_token = final_norm_.backward(grad_norm_output);
    grad_cls_token.print("grad_cls_token");

    // Crear tensor de gradientes completo
    Tensor grad_sequence_full(encoder_output_shape_cache_);
    grad_sequence_full.zero_data();
    std::cout << "[ViT::backward] grad_sequence_full (cero) shape: ";
    grad_sequence_full.print("grad_sequence_full (vacío)");

    // Insertar gradiente del CLS token
    std::cout << "[ViT::backward] -> set_row(0, grad_cls_token)..." << std::endl;
    grad_sequence_full.set_row(0, grad_cls_token);

    // Backward a través de la pila de Encoders
    for (int i = encoder_layers_.size() - 1; i >= 0; --i)
    {
        std::cout << "[ViT::backward] -> encoder_layers_[" << i << "].backward..." << std::endl;
        grad_sequence_full = encoder_layers_[i].backward(grad_sequence_full);
        grad_sequence_full.print("grad_sequence_full (post encoder " + std::to_string(i) + ")");
    }

    std::cout << "[ViT::backward] -> patch_embedding_.backward..." << std::endl;
    patch_embedding_.backward(grad_sequence_full);
}

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