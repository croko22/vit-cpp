#pragma once

#include "patch_embedding.hpp"
#include "encoder.hpp"
#include "../core/tensor.hpp"
#include <vector>

class VisionTransformer
{
private:
    PatchEmbedding patch_embedding_;
    std::vector<EncoderLayer> encoder_layers_;
    LayerNormalization final_norm_;
    Tensor classification_head_w_;
    Tensor classification_head_b_;

    // --- backward cache ---
    Tensor cls_token_output_cache_;
    Tensor normalized_output_cache_;              // <--- AÑADIDO para el error 1
    std::vector<int> encoder_output_shape_cache_; // <--- AÑADIDO para el error 2

public:
    VisionTransformer(int image_size, int patch_size, int in_channels, int num_classes,
                      int d_model, int num_heads, int d_ff, int num_layers);
    Tensor forward(const Tensor &image);
    void backward(const Tensor &grad_loss);

    void get_parameters(std::vector<Tensor *> &params);

    void zero_grad();

    void get_classification_head_parameters(std::vector<Tensor *> &params);
};