#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include "../../include/core/tensor.h"
#include "../../include/core/activation.h"
#include "../../include/core/random.h"
#include "linear.h"
#include "layernorm.h"
#include "encoder.h"
#include <vector>
#include <memory>
#include <cmath>
#include <numeric>

class VisionTransformer
{
public:
    int image_size, patch_size, d_model, num_layers, num_classes;
    int num_patches;
    Linear patch_embedding;
    Tensor class_token, position_embeddings;
    std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks;
    Linear classification_head;
    LayerNorm final_ln;

    Tensor last_patches;
    Tensor last_logits;

    VisionTransformer(int img_size, int patch_sz, int d_mod, int n_layers, int n_classes);

    Tensor image_to_patches(const Tensor &image);
    Tensor forward(const Tensor &image);
    void backward(int true_label);
    float compute_loss(const Tensor &logits, int true_label);
    void update_weights(float lr, int batch_size = 1);
    void zero_grad();
    int predict(const Tensor &image);
    int predictWithLogits(const Tensor &logits);
    void load_model(const std::string &filename);
    void save_model(const std::string &filename) const;
};

#endif
