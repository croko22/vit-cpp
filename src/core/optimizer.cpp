#include "../../include/core/optimizer.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

SGD::SGD(std::vector<Tensor *> &parameters, float learning_rate)
    : parameters_(parameters), learning_rate_(learning_rate) {}

void SGD::step()
{
    for (size_t i = 0; i < parameters_.size(); ++i)
    {
        Tensor *param = parameters_[i];
        if (param && param->grad_)
        {
            std::cout << "[SGD::step] Param[" << i << "] shape: ";
            param->print("param");

            std::cout << "[SGD::step] Grad[" << i << "] shape: ";
            param->grad_->print("grad");

            Tensor scaled_grad = *(param->grad_) * learning_rate_;
            std::cout << "[SGD::step] Scaled grad shape: ";
            scaled_grad.print("scaled_grad");

            if (param->get_shape() != scaled_grad.get_shape())
            {
                std::cerr << "SHAPE MISMATCH:\nParam: ";
                param->print("param");
                std::cerr << "Grad: ";
                param->grad_->print("grad");
                throw std::runtime_error("Shape mismatch en SGD::step()");
            }

            try
            {
                *param = *param - scaled_grad;
            }
            catch (const std::exception &e)
            {
                std::cerr << "\n!!! EXCEPCIÓN DURANTE LA RESTA EN SGD::step !!!\n";
                std::cerr << "Tensor[" << i << "] shape: ";
                param->print("param");
                std::cerr << "Grad[" << i << "] shape: ";
                param->grad_->print("grad");
                std::cerr << "scaled_grad shape: ";
                scaled_grad.print("scaled_grad");
                std::cerr << "Error: " << e.what() << std::endl;
                throw;
            }
        }
        else
        {
            std::cerr << "[SGD::step] Param[" << i << "] o su gradiente es nulo\n";
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