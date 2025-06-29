#include "../include/model/decoder.hpp"
#include "../include/core/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

Tensor create_look_ahead_mask(int size)
{
    Tensor mask({size, size});
    float *data = mask.get_data();
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            if (j > i)
            {
                data[i * size + j] = -1e9f;
            }
            else
            {
                data[i * size + j] = 0.0f;
            }
        }
    }
    return mask;
}

int main()
{
    std::cout << "=== Transformer Decoder Layer Example ===" << std::endl;

    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int src_seq_len = 8;
    int tgt_seq_len = 6;

    std::cout << "Architecture: d_model=" << d_model
              << ", num_heads=" << num_heads
              << ", d_ff=" << d_ff << std::endl;
    std::cout << "Sequences: src_len=" << src_seq_len
              << ", tgt_len=" << tgt_seq_len << std::endl;

    DecoderLayer decoder_layer(d_model, num_heads, d_ff);

    std::vector<int> encoder_shape = {src_seq_len, d_model};
    Tensor encoder_output(encoder_shape);

    std::vector<float> encoder_data(src_seq_len * d_model);
    for (int i = 0; i < src_seq_len; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {

            float context_pattern = 0.4f * std::sin(0.03f * j + i * 0.7f);
            float positional = 0.2f * std::cos(0.01f * j * (i + 1));
            encoder_data[i * d_model + j] = context_pattern + positional;
        }
    }
    encoder_output.from_vector(encoder_data);

    std::vector<int> target_shape = {tgt_seq_len, d_model};
    Tensor target_input(target_shape);

    std::vector<float> target_data(tgt_seq_len * d_model);
    for (int i = 0; i < tgt_seq_len; ++i)
    {
        for (int j = 0; j < d_model; ++j)
        {

            float target_pattern = 0.3f * std::sin(0.04f * j + i * 0.6f);
            float position_encoding = 0.15f * std::cos(0.02f * j * (i + 1));
            target_data[i * d_model + j] = target_pattern + position_encoding + 0.1f;
        }
    }
    target_input.from_vector(target_data);

    Tensor look_ahead_mask = create_look_ahead_mask(tgt_seq_len);

    std::cout << "\n=== INPUTS ===" << std::endl;
    std::cout << "Encoder output shape: [" << src_seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Target input shape: [" << tgt_seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Look-ahead mask shape: [" << tgt_seq_len << ", " << tgt_seq_len << "]" << std::endl;

    std::cout << "\nLook-ahead mask (first 6x6):" << std::endl;
    std::vector<float> mask_data = look_ahead_mask.to_vector();
    for (int i = 0; i < std::min(6, tgt_seq_len); ++i)
    {
        for (int j = 0; j < std::min(6, tgt_seq_len); ++j)
        {
            if (mask_data[i * tgt_seq_len + j] < -1e8)
            {
                std::cout << " -∞ ";
            }
            else
            {
                std::cout << "  0 ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "\nTarget input sample (first 10 values of each position):" << std::endl;
    std::vector<float> target_vec = target_input.to_vector();
    for (int i = 0; i < std::min(4, tgt_seq_len); ++i)
    {
        std::cout << "Pos " << i << ": ";
        for (int j = 0; j < 10; ++j)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << target_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    std::cout << "\nEncoder output sample (first 10 values of each position):" << std::endl;
    std::vector<float> encoder_vec = encoder_output.to_vector();
    for (int i = 0; i < std::min(4, src_seq_len); ++i)
    {
        std::cout << "Pos " << i << ": ";
        for (int j = 0; j < 10; ++j)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << encoder_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    std::cout << "\n=== PROCESSING ===" << std::endl;
    std::cout << "Applying Decoder Layer..." << std::endl;
    std::cout << "Process: Target → Masked Self-Attention → Add&Norm → " << std::endl;
    std::cout << "         Cross-Attention(Query=Target, Key=Encoder, Value=Encoder) → Add&Norm →" << std::endl;
    std::cout << "         FFN → Add&Norm → Output" << std::endl;

    Tensor output = decoder_layer.forward(target_input, encoder_output, &look_ahead_mask, nullptr);

    std::cout << "\n=== OUTPUT ===" << std::endl;
    std::cout << "Output tensor shape: [" << output.get_shape()[0]
              << ", " << output.get_shape()[1] << "]" << std::endl;

    std::cout << "Output sample (first 10 values of each position):" << std::endl;
    std::vector<float> output_vec = output.to_vector();
    for (int i = 0; i < std::min(4, tgt_seq_len); ++i)
    {
        std::cout << "Pos " << i << ": ";
        for (int j = 0; j < 10; ++j)
        {
            std::cout << std::fixed << std::setprecision(3)
                      << output_vec[i * d_model + j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    const auto &output_shape = output.get_shape();
    bool shape_correct = (output_shape[0] == tgt_seq_len && output_shape[1] == d_model);

    bool has_non_zero = false;
    for (int i = 0; i < 100 && !has_non_zero; ++i)
    {
        if (std::abs(output_vec[i]) > 1e-6)
            has_non_zero = true;
    }

    bool is_different = false;
    for (int i = 0; i < 100 && !is_different; ++i)
    {
        if (std::abs(output_vec[i] - target_vec[i]) > 1e-4)
            is_different = true;
    }

    std::cout << "\n=== VERIFICATION ===" << std::endl;
    std::cout << "Shape preservation: " << (shape_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Non-zero outputs: " << (has_non_zero ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Output differs from input: " << (is_different ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Expected shape: [" << tgt_seq_len << ", " << d_model << "]" << std::endl;
    std::cout << "Actual shape: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;

    float target_mean = 0.0f, output_mean = 0.0f;
    for (float val : target_vec)
        target_mean += val;
    for (float val : output_vec)
        output_mean += val;
    target_mean /= target_vec.size();
    output_mean /= output_vec.size();

    std::cout << "\n=== STATISTICS ===" << std::endl;
    std::cout << "Target input mean: " << std::fixed << std::setprecision(6) << target_mean << std::endl;
    std::cout << "Output mean: " << output_mean << std::endl;
    std::cout << "\nDecoder layer successfully processes target sequence using encoder context!" << std::endl;

    return 0;
}