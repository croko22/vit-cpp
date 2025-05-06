#include <iostream>
#include "attention.cuh"

int main()
{
    const int N = 4;
    float h_Q[N * N] = {1, 2, 3, 4,
                        4, 3, 2, 1,
                        1, 0, 0, 1,
                        2, 2, 2, 2};
    float h_K[N * N] = {1, 0, 0, 1,
                        0, 1, 1, 0,
                        1, 1, 0, 0,
                        0, 0, 1, 1};
    float h_V[N * N] = {1, 0, 2, 0,
                        0, 1, 0, 2,
                        2, 0, 1, 0,
                        0, 2, 0, 1};
    float h_output[N * N];

    run_attention(h_Q, h_K, h_V, h_output, N);

    std::cout << "Attention Output:\n";
    for (int i = 0; i < N * N; ++i)
    {
        std::cout << h_output[i] << " ";
        if ((i + 1) % N == 0)
            std::cout << "\n";
    }

    return 0;
}
