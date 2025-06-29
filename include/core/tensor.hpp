#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>

enum class Device
{
    CPU
};

class Tensor
{
private:
    std::vector<int> shape_;
    Device device_;
    float *data_ = nullptr;
    size_t size_ = 0;

    void allocate();
    void deallocate();

public:
    Tensor(const std::vector<int> &shape);
    ~Tensor();

    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(float scalar) const;
    friend Tensor operator*(float scalar, const Tensor &tensor);
    Tensor transpose() const;

    void from_vector(const std::vector<float> &host_data);
    std::vector<float> to_vector() const;

    float *get_data() const { return data_; }
    size_t get_size() const { return size_; }
    const std::vector<int> &get_shape() const { return shape_; }
    int get_dims() const { return shape_.size(); }

    void print() const;
};

#endif