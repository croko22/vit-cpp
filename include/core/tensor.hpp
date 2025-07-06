#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <string>

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
    float *grad_data_ = nullptr;
    size_t size_ = 0;

    void allocate();
    void deallocate();
    void allocate_grad();
    void deallocate_grad();

public:
    std::shared_ptr<Tensor> grad_ = nullptr;

    Tensor *input1_for_grad = nullptr;
    Tensor *input2_for_grad = nullptr;
    std::string op_type_for_grad;

    Tensor(); // Constructor por defecto, crea un tensor de tamaño 1
    Tensor(const std::vector<int> &shape);
    ~Tensor();

    Tensor(Tensor &&other) noexcept;
    Tensor(const Tensor &other);            // Constructor de copia
    Tensor &operator=(const Tensor &other); // Asignación de copia

    Tensor &operator=(Tensor &&other) noexcept;
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(float scalar) const;
    friend Tensor operator*(float scalar, const Tensor &tensor);
    Tensor transpose() const;

    void from_vector(const std::vector<float> &host_data);
    std::vector<float> to_vector() const;

    void init_grad();
    float *get_data() const { return data_; }
    float *get_grad() const { return grad_data_; }
    void add_grad(const Tensor &grad_tensor);
    size_t get_size() const { return size_; }
    const std::vector<int> &get_shape() const { return shape_; }
    int get_dims() const { return shape_.size(); }
    // --- NUEVAS UTILIDADES ---
    Tensor get_row(int row_index) const;
    void set_row(int row_index, const Tensor &row_tensor);
    void zero_grad();
    void zero_data();
    void print(const std::string &title = "Tensor") const;
};

#endif