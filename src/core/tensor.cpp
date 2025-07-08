#include "../../include/core/tensor.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <iomanip>

Tensor::Tensor()
    : shape_({1}), device_(Device::CPU), data_(nullptr), grad_data_(nullptr), size_(1)
{
    allocate();
}

Tensor::Tensor(const std::vector<int> &shape)
    : shape_(shape)
{
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    allocate();
}

Tensor::~Tensor()
{
    deallocate();
}

Tensor::Tensor(Tensor &&other) noexcept
    : shape_(std::move(other.shape_)), device_(Device::CPU), data_(other.data_), size_(other.size_)
{
    other.data_ = nullptr;
    other.grad_data_ = nullptr;
    other.size_ = 0;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        deallocate();
        shape_ = std::move(other.shape_);
        device_ = Device::CPU;
        data_ = other.data_;
        grad_data_ = other.grad_data_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    const auto &other_shape = other.get_shape();
    if (this->shape_ == other_shape)
    {
        Tensor result(this->shape_);
        for (size_t i = 0; i < this->size_; ++i)
            result.get_data()[i] = this->data_[i] + other.get_data()[i];
        return result;
    }
    if (this->shape_.size() == 2 && other_shape.size() == 2 && this->shape_[1] == other_shape[1] && other_shape[0] == 1)
    {
        Tensor result(this->shape_);
        int rows = this->shape_[0], cols = this->shape_[1];
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                result.get_data()[i * cols + j] = this->data_[i * cols + j] + other.get_data()[j];
            }
        }
        return result;
    }
    throw std::invalid_argument("Shape mismatch in addition or unsupported broadcasting");
}

Tensor Tensor::operator-(const Tensor &other) const
{
    if (shape_ != other.shape_)
        throw std::invalid_argument("Shape mismatch in subtraction");
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i)
        result.data_[i] = data_[i] - other.data_[i];
    return result;
}

Tensor Tensor::operator*(float scalar) const
{
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i)
        result.data_[i] = data_[i] * scalar;
    return result;
}

Tensor operator*(float scalar, const Tensor &tensor)
{
    return tensor * scalar;
}

// Tensor Tensor::transpose() const
// {
//     if (shape_.size() != 2)
//         throw std::runtime_error("Transpose only supports 2D tensors");
//     int rows = shape_[0], cols = shape_[1];
//     Tensor result({cols, rows});
//     for (int r = 0; r < rows; ++r)
//         for (int c = 0; c < cols; ++c)
//             result.data_[c * rows + r] = data_[r * cols + c];
//     return result;
// }

Tensor Tensor::transpose() const
{
    if (this->get_dims() < 2)
    {
        throw std::runtime_error("transpose(): tensor must have >= 2 dimensions, but got " + std::to_string(this->get_dims()));
    }
    int rows = shape_[0], cols = shape_[1];
    Tensor result({cols, rows});
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            result.get_data()[c * rows + r] = this->data_[r * cols + c];
        }
    }
    return result;
}

void Tensor::allocate()
{
    if (size_ == 0)
        return;
    data_ = new float[size_];
}

void Tensor::deallocate()
{
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
}

void Tensor::from_vector(const std::vector<float> &host_data)
{
    if (host_data.size() != size_)
    {
        throw std::invalid_argument("Input vector size does not match tensor size.");
    }
    std::copy(host_data.begin(), host_data.end(), data_);
}

std::vector<float> Tensor::to_vector() const
{
    std::vector<float> host_vector(size_);
    std::copy(data_, data_ + size_, host_vector.begin());
    return host_vector;
}

void Tensor::allocate_grad()
{
    if (grad_data_)
        delete[] grad_data_;
    grad_data_ = new float[size_];
    zero_grad(); // Inicializa a cero
}

void Tensor::deallocate_grad()
{
    delete[] grad_data_;
    grad_data_ = nullptr;
}

void Tensor::zero_grad()
{
    if (grad_data_)
        memset(grad_data_, 0, size_ * sizeof(float));
}

void Tensor::add_grad(const Tensor &grad_tensor)
{
    if (grad_data_ == nullptr)
        allocate_grad();

    size_t total = size_;
    for (size_t i = 0; i < total; ++i)
    {
        grad_data_[i] += grad_tensor.get_data()[i];
    }
}

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), size_(other.size_)
{
    allocate();
    std::copy(other.data_, other.data_ + size_, data_);
    if (other.grad_)
    {
        grad_ = std::make_shared<Tensor>(*other.grad_);
    }
}

Tensor &Tensor::operator=(const Tensor &other)
{
    if (this != &other)
    {
        deallocate();
        shape_ = other.shape_;
        size_ = other.size_;
        allocate();
        std::copy(other.data_, other.data_ + size_, data_);
        if (other.grad_)
        {
            grad_ = std::make_shared<Tensor>(*other.grad_);
        }
        else
        {
            grad_ = nullptr;
        }
    }
    return *this;
}

void Tensor::init_grad()
{
    if (!grad_)
    {
        grad_ = std::make_shared<Tensor>(shape_);
    }
    grad_->zero_grad(); // Asegura que el gradiente del gradiente también esté limpio si fuera necesario
}

void Tensor::print(const std::string &title) const
{
    auto vec = to_vector();
    std::cout << title << " (Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
    }
    std::cout << "])\n";

    int rows = (get_dims() > 1) ? shape_[0] : 1;
    int cols = (get_dims() > 0) ? shape_.back() : 0;
    for (int r = 0; r < std::min(rows, 4); ++r)
    {
        for (int c = 0; c < std::min(cols, 8); ++c)
        {
            std::cout << std::fixed << std::setprecision(4) << vec[r * cols + c] << "\t";
        }
        std::cout << (cols > 8 ? "...\n" : "\n");
    }
    if (rows > 4)
        std::cout << "...\n";
}

void Tensor::zero_data()
{
    if (data_)
        memset(data_, 0, size_ * sizeof(float));
}

Tensor Tensor::get_row(int row_index) const
{
    if (this->get_dims() != 2)
        throw std::runtime_error("get_row solo funciona en tensores 2D");
    int cols = this->get_shape()[1];
    Tensor row({1, cols});
    const float *start_ptr = this->data_ + row_index * cols;
    std::copy(start_ptr, start_ptr + cols, row.get_data());
    return row;
}

void Tensor::set_row(int row_index, const Tensor &row_tensor)
{
    if (this->get_dims() != 2 || row_tensor.get_dims() != 2)
        throw std::runtime_error("set_row solo funciona en tensores 2D");
    int this_cols = this->get_shape()[1];
    int row_cols = row_tensor.get_shape()[1];
    if (this_cols != row_cols || row_tensor.get_shape()[0] != 1)
        throw std::runtime_error("Las formas para set_row no coinciden");

    float *start_ptr = this->data_ + row_index * this_cols;
    std::copy(row_tensor.get_data(), row_tensor.get_data() + row_cols, start_ptr);
}