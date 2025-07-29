#include <iostream>
#include <cassert>
#include <cmath>

// Incluir la versión apropiada según la compilación
#ifdef USE_CUDA
    #include "../src/core/cuda/tensor.cu"
   // using TensorType = Tensor;
    #define TENSOR_BACKEND "CUDA/GPU"
#else
    #include "../src/core/tensor.cpp"
    #define TENSOR_BACKEND "CPU"
#endif
using TensorType = Tensor;

void print_backend_info() {
    std::cout << "🔧 Usando backend: " << TENSOR_BACKEND << std::endl;
    std::cout << "===========================================" << std::endl;
}

void test_constructor_and_access() {
    std::cout << "Test: Constructor y Acceso... ";
    
#ifdef USE_CUDA
    // Para CUDA, usar vectores para inicializar
    std::vector<float> data = {1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f};
    TensorType t({2, 3}, data);
    
    // Verificar usando get_element para CUDA
    assert(std::abs(t.get_element(0, 0) - 1.0f) < 1e-6);
    assert(std::abs(t.get_element(0, 1) - 2.0f) < 1e-6);
    assert(std::abs(t.get_element(1, 2) - 3.0f) < 1e-6);
#else
    // Para CPU, usar acceso directo
    TensorType t({2, 3});
    t(0, 0) = 1.0f;
    t(0, 1) = 2.0f;
    t(1, 2) = 3.0f;
    
    assert(t(0, 0) == 1.0f);
    assert(t(0, 1) == 2.0f);
    assert(t(1, 2) == 3.0f);
#endif
    
    std::cout << "✅" << std::endl;
}

void test_addition() {
    std::cout << "Test: Suma... ";
    
#ifdef USE_CUDA
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {3.0f, 4.0f, 5.0f, 6.0f};
    TensorType a({2, 2}, data_a);
    TensorType b({2, 2}, data_b);
    
    TensorType c = a + b;
    assert(std::abs(c.get_element(0, 0) - 4.0f) < 1e-6);
    assert(std::abs(c.get_element(0, 1) - 6.0f) < 1e-6);
#else
    TensorType a({2, 2});
    TensorType b({2, 2});
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = 3; a(1, 1) = 4;
    b(0, 0) = 3; b(0, 1) = 4;
    b(1, 0) = 5; b(1, 1) = 6;
    
    TensorType c = a + b;
    assert(c(0, 0) == 4);
    assert(c(0, 1) == 6);
#endif
    
    std::cout << "✅" << std::endl;
}

void test_multiplication() {
    std::cout << "Test: Multiplicación de matrices... ";
    
#ifdef USE_CUDA
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> data_b = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    TensorType a({2, 3}, data_a);
    TensorType b({3, 2}, data_b);
    
    TensorType c = a * b;
    auto shape = c.get_shape();
    assert(shape[0] == 2 && shape[1] == 2);
    
    // 1*4 + 2*6 + 3*8 = 4 + 12 + 24 = 40
    float expected = 1*4 + 2*6 + 3*8;
    assert(std::abs(c.get_element(0, 0) - expected) < 1e-4);
#else
    TensorType a({2, 3});
    TensorType b({3, 2});
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
    b(0, 0) = 4; b(0, 1) = 7;
    b(1, 0) = 5; b(1, 1) = 8;
    b(2, 0) = 6; b(2, 1) = 9;
    
    TensorType c = a * b;
    auto shape = c.get_shape();
    assert(shape[0] == 2 && shape[1] == 2);
    assert(c(0, 0) == 1*4 + 2*5 + 3*6); // 32
#endif
    
    std::cout << "✅" << std::endl;
}

void test_transpose() {
    std::cout << "Test: Transposición... ";
    
#ifdef USE_CUDA
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    TensorType t({2, 3}, data);
    
    TensorType tr = t.transpose();
    auto shape = tr.get_shape();
    assert(shape[0] == 3 && shape[1] == 2);
    assert(std::abs(tr.get_element(0, 0) - 1.0f) < 1e-6);
    assert(std::abs(tr.get_element(1, 0) - 2.0f) < 1e-6);
    assert(std::abs(tr.get_element(2, 0) - 3.0f) < 1e-6);
#else
    TensorType t({2, 3});
    t(0, 0) = 1; t(0, 1) = 2; t(0, 2) = 3;
    t(1, 0) = 4; t(1, 1) = 5; t(1, 2) = 6;
    
    TensorType tr = t.transpose();
    auto shape = tr.get_shape();
    assert(shape[0] == 3 && shape[1] == 2);
    assert(tr(0, 0) == 1);
    assert(tr(1, 0) == 2);
    assert(tr(2, 0) == 3);
#endif
    
    std::cout << "✅" << std::endl;
}

void test_xavier_init() {
    std::cout << "Test: Inicialización Xavier... ";
    
    TensorType t({100, 100});
    TensorType::xavier_init(t);
    float n = t.norm();
    assert(n > 0);
    
    std::cout << "✅ (norm: " << n << ")" << std::endl;
}

void test_scalar_multiplication() {
    std::cout << "Test: Multiplicación por escalar... ";
    
#ifdef USE_CUDA
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    TensorType t({2, 2}, data);
    
    TensorType result = t * 2.0f;
    assert(std::abs(result.get_element(0, 0) - 2.0f) < 1e-6);
    assert(std::abs(result.get_element(0, 1) - 4.0f) < 1e-6);
#else
    TensorType t({2, 2});
    t(0, 0) = 1; t(0, 1) = 2;
    t(1, 0) = 3; t(1, 1) = 4;
    
    TensorType result = t * 2.0f;
    assert(result(0, 0) == 2);
    assert(result(0, 1) == 4);
#endif
    
    std::cout << "✅" << std::endl;
}

void test_argmax() {
    std::cout << "Test: Argmax... ";
    
#ifdef USE_CUDA
    std::vector<float> data = {0.1f, 0.3f, 0.7f, 0.5f, 0.2f};
    TensorType t({1, 5}, data);
#else
    TensorType t({1, 5});
    t(0, 0) = 0.1f; t(0, 1) = 0.3f; t(0, 2) = 0.7f; 
    t(0, 3) = 0.5f; t(0, 4) = 0.2f;
#endif
    
    int max_idx = t.argmax();
    assert(max_idx == 2); // El valor más alto (0.7) está en el índice 2
    
    std::cout << "✅" << std::endl;
}

void benchmark_matrix_multiplication() {
    std::cout << "\n🚀 Benchmark: Multiplicación de matrices grandes" << std::endl;
    
    const int size = 512;
    
    auto start = std::chrono::high_resolution_clock::now();
    
#ifdef USE_CUDA
    // Inicializar con datos aleatorios
    std::vector<float> data_a(size * size, 1.0f);
    std::vector<float> data_b(size * size, 1.0f);
    
    TensorType a({size, size}, data_a);
    TensorType b({size, size}, data_b);
    
    TensorType::xavier_init(a);
    TensorType::xavier_init(b);
    
    TensorType c = a * b;
    TensorType::sync(); // Sincronizar GPU
#else
    TensorType a({size, size});
    TensorType b({size, size});
    
    TensorType::xavier_init(a);
    TensorType::xavier_init(b);
    
    TensorType c = a * b;
#endif
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "⏱️  Multiplicación " << size << "x" << size 
              << " completada en " << duration.count() << " ms" << std::endl;
    std::cout << "📊 Rendimiento: " << TENSOR_BACKEND << std::endl;
}

void run_all_tests() {
    print_backend_info();
    
    try {
        test_constructor_and_access();
        test_addition();
        test_scalar_multiplication();
        test_multiplication();
        test_transpose();
        test_xavier_init();
        test_argmax();
        
        std::cout << "\n✅ Todos los tests básicos pasaron correctamente!" << std::endl;
        
        benchmark_matrix_multiplication();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error en los tests: " << e.what() << std::endl;
        throw;
    }
    
//#ifdef USE_CUDA
//    // Limpiar recursos CUDA si es necesario
//    TensorType::cleanup();
//#endif
}

int main() {
    run_all_tests();
    std::cout << "\n🎉 Programa completado exitosamente!" << std::endl;
    return 0;
    
}