#include "../../include/core/optimizer.hpp"

SGD::SGD(std::vector<Tensor *> &parameters, float learning_rate)
    : parameters_(parameters), learning_rate_(learning_rate) {}

void SGD::step()
{
    for (Tensor *param : parameters_)
    {
        if (param && param->grad_)
        {
            // La actualización de pesos: W = W - lr * W.grad
            *param = *param - (*(param->grad_) * learning_rate_);
        }
    }
}

void SGD::zero_grad()
{
    for (Tensor *param : parameters_)
    {
        if (param && param->grad_)
        {
            param->grad_->zero_grad();
        }
    }
}