#include "../include/model/layernorm.hpp"
#include "../include/core/tensor.hpp"
#include <iostream>
#include <cmath>

int main()
{
    int feature_size = 4;
    float epsilon = 1e-5;
    LayerNormalization layer_norm(feature_size, epsilon);

    std::vector<int> shape = {2, 4};
    Tensor input(shape);

    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 1.5f, 2.5f, 3.5f};

    input.from_vector(input_data);

    std::cout << "Input tensor:" << std::endl;
    input.print();

    Tensor output = layer_norm.forward(input);

    std::cout << "\nOutput after LayerNorm:" << std::endl;
    output.print();

    std::cout << "\nVerification (each row should have mean ≈ 0, std ≈ 1):" << std::endl;

    std::vector<float> output_data = output.to_vector();
    const auto &output_shape = output.get_shape();
    int rows = output_shape[0];
    int cols = output_shape[1];

    for (int i = 0; i < rows; ++i)
    {
        float mean = 0.0f, variance = 0.0f;

        for (int j = 0; j < cols; ++j)
        {
            mean += output_data[i * cols + j];
        }
        mean /= cols;

        for (int j = 0; j < cols; ++j)
        {
            float diff = output_data[i * cols + j] - mean;
            variance += diff * diff;
        }
        variance /= cols;

        std::cout << "Row " << i << ": mean = " << mean
                  << ", std = " << std::sqrt(variance) << std::endl;
    }

    return 0;
}