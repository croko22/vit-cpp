#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include "../../include/core/tensor.h"
#include "../../include/core/activation.h" // For Activation::softmax
#include "../../include/core/random.h"     // For Random::randn
#include "linear.h"
#include "layernorm.h"
#include "encoder.h" // VisionTransformer uses TransformerBlock
#include <vector>    // For std::vector
#include <memory>    // For std::unique_ptr
#include <cmath>     // For log, max
#include <numeric>   // For iota (though not directly used in VT, good to have for related utilities)

// Vision Transformer mejorado
class VisionTransformer
{
public:
    int patch_size, d_model, num_layers, num_classes;
    int image_size, num_patches;
    Linear patch_embedding;
    Tensor class_token, position_embeddings;
    std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks;
    Linear classification_head;
    LayerNorm final_ln;

    // For backpropagation: store intermediate results
    Tensor last_patches;
    Tensor last_logits; // Store the final logits for loss calculation and backward pass

    VisionTransformer(int img_size, int patch_sz, int d_mod, int n_layers, int n_classes);

    Tensor image_to_patches(const Tensor &image);
    Tensor forward(const Tensor &image);
    void backward(int true_label);
    float compute_loss(const Tensor &logits, int true_label);
    void update_weights(float lr);
    void zero_grad();
    int predict(const Tensor &image);
    void load_model(const std::string &filename);
    void save_model(const std::string &filename) const;
};

#endif // VISION_TRANSFORMER_H
