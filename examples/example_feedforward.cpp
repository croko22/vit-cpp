#include "../include/model/feedforward.hpp"
#include <iostream>

int main()
{
    int d_model = 4;
    int d_ff = 8;
    float dropout = 0.1f;

    FeedForwardNetwork ffn(d_model, d_ff, dropout);

    std::vector<std::vector<float>> input = {
        {0.5f, -0.2f, 0.1f, 0.3f},
        {0.0f, 0.8f, -0.5f, 0.9f}};

    auto output = ffn.forward(input);

    std::cout << "Output:\n";
    for (const auto &row : output)
    {
        for (float val : row)
            std::cout << val << " ";
        std::cout << "\n";
    }

    return 0;
}
