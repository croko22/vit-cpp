#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "encoder.hpp"
#include "../core/tensor.hpp"
#include <vector>

class Transformer
{
private:
    int num_layers_;
    int d_model_;
    std::vector<EncoderLayer> encoder_layers_;
    Tensor final_linear_layer_weights_;
    Tensor decoder_output;

public:
    Transformer(int num_layers, int d_model, int num_heads, int d_ff, int target_vocab_size);
    Tensor forward(const Tensor &src_tensor, const Tensor &tgt_tensor);
    void backward(const Tensor &grad_output);
    void get_parameters(std::vector<Tensor *> &params);

private:
    Tensor encode(const Tensor &src_tensor);
    static Tensor create_look_ahead_mask(int size);
};

#endif // TRANSFORMER_HPP