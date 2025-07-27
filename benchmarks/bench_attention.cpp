#include "../include/model/multihead_attention.h"
#include "../include/core/tensor.h"
#include <chrono>
#include <iostream>

void benchmark_attention()
{
    std::cout << "=== Benchmark Multi-Head Attention ===" << std::endl;

    std::vector<int> seq_lengths = {16, 32, 64, 128};
    std::vector<int> d_models = {64, 128, 256};
    std::vector<int> num_heads = {4, 8, 12};

    for (int seq_len : seq_lengths)
    {
        for (int d_model : d_models)
        {
            for (int heads : num_heads)
            {
                if (d_model % heads != 0)
                    continue;

                std::cout << "\nTesting: seq=" << seq_len
                          << ", d_model=" << d_model
                          << ", heads=" << heads << std::endl;

                MultiHeadAttention mha(d_model, heads);
                Tensor input({seq_len, d_model});

                for (int i = 0; i < input.get_data().size(); i++)
                {
                    input.get_data()[i] = static_cast<float>(rand()) / RAND_MAX;
                }

                auto start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 100; i++)
                {
                    Tensor output = mha.forward(input);
                }
                auto end = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Forward: " << duration.count() / 100.0 << " μs/iteration" << std::endl;

                Tensor grad_output({seq_len, d_model});
                for (int i = 0; i < grad_output.get_data().size(); i++)
                {
                    grad_output.get_data()[i] = 1.0f;
                }

                start = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 100; i++)
                {
                    mha.forward(input);
                    Tensor grad_input = mha.backward(grad_output);
                }
                end = std::chrono::high_resolution_clock::now();

                duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "  Backward: " << duration.count() / 100.0 << " μs/iteration" << std::endl;
            }
        }
    }
}

int main()
{
    benchmark_attention();
    return 0;
}