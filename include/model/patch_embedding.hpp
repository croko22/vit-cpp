#pragma once

#include "../core/tensor.hpp"

class PatchEmbedding
{
public:
    Tensor projection;
    Tensor cls_token;
    Tensor position_embeddings;

private:
    int patch_size_;
    int d_model_;
    int num_patches_;

public:
    PatchEmbedding(int image_size, int patch_size, int in_channels, int d_model);
    Tensor forward(const Tensor &image);
    void backward(const Tensor &grad_output);
    void get_parameters(std::vector<Tensor *> &params);
};