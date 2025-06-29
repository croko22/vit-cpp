#include "../../include/core/tensor.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>

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
        size_ = other.size_;

        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape_ != other.shape_)
        throw std::invalid_argument("Shape mismatch in addition");
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i)
        result.data_[i] = data_[i] + other.data_[i];
    return result;
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

Tensor Tensor::transpose() const
{
    if (shape_.size() != 2)
        throw std::runtime_error("Transpose only supports 2D tensors");
    int rows = shape_[0], cols = shape_[1];
    Tensor result({cols, rows});
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            result.data_[c * rows + r] = data_[r * cols + c];
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

void Tensor::print() const
{
    auto vec = to_vector();
    std::cout << "Tensor (Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i)
    {
        std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
    }
    std::cout << "], Device: CPU)\n";

    int rows = (get_dims() > 1) ? shape_[0] : 1;
    int cols = (get_dims() > 0) ? shape_.back() : 0;
    for (int r = 0; r < std::min(rows, 10); ++r)
    {
        for (int c = 0; c < std::min(cols, 10); ++c)
        {
            std::cout << vec[r * cols + c] << " ";
        }
        std::cout << (cols > 10 ? "...\n" : "\n");
    }
    if (rows > 10)
        std::cout << "...\n";
}