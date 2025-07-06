#pragma once

#include "tensor.hpp"
#include <vector>

class SGD
{
private:
    std::vector<Tensor *> parameters_;
    float learning_rate_;

public:
    SGD(std::vector<Tensor *> &parameters, float learning_rate = 0.01f);

    // Actualiza todos los pesos usando sus gradientes
    void step();

    // Pone a cero los gradientes de todos los parámetros
    void zero_grad();
};