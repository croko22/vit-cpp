#include "../../include/optimizer/sgd.h"

SGD::SGD(std::vector<Parameter> params, float lr)
    : Optimizer(params, lr) {}

void SGD::step()
{
    for (const auto &p : params)
    {
        for (int i = 0; i < p.weight->get_size(); ++i)
        {
            p.weight->data[i] -= lr * p.grad->data[i];
        }
    }
}