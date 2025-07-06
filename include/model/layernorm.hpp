#ifndef LAYERNORM_HPP
#define LAYERNORM_HPP

#include "../core/tensor.hpp"

class LayerNormalization
{
public:
    // Parámetros aprendibles
    Tensor gamma_; // scale
    Tensor beta_;  // shift
private:
    float epsilon_ = 1e-5;

    int feature_size_;

    // Cache para backward pass
    Tensor input_cache_;
    Tensor mean_cache_;
    Tensor var_cache_;
    Tensor normalized_input_cache_;

public:
    LayerNormalization(int feature_size, float epsilon = 1e-5);
    Tensor forward(const Tensor &input);
    Tensor backward(const Tensor &grad_output);
    void get_parameters(std::vector<Tensor *> &params);
    void zero_all_grads();
};

#endif // LAYERNORM_HPP