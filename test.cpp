#include <iostream>
#include <cassert>
#include "src/core/cuda/tensor.cu"
#include "src/core/tensor.cpp"

void test_constructor_and_access() {
    Tensor t(2, 3);
    t(0, 0) = 1.0f;
    t(0, 1) = 2.0f;
    t(1, 2) = 3.0f;
    assert(t(0, 0) == 1.0f);
    assert(t(0, 1) == 2.0f);
    assert(t(1, 2) == 3.0f);
}

void test_addition() {
    Tensor a(2, 2);
    Tensor b(2, 2);
    a(0, 0) = 1; a(0, 1) = 2;
    b(0, 0) = 3; b(0, 1) = 4;
    Tensor c = a + b;
    assert(c(0, 0) == 4);
    assert(c(0, 1) == 6);
}

void test_multiplication() {
    Tensor a(2, 3);
    Tensor b(3, 2);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    b(0, 0) = 4; b(1, 0) = 5; b(2, 0) = 6;
    Tensor c = a * b;
    assert(c.rows == 2 && c.cols == 2);
    assert(c(0, 0) == 1*4 + 2*5 + 3*6);
}

void test_transpose() {
    Tensor t(2, 3);
    t(0, 0) = 1; t(0, 1) = 2; t(0, 2) = 3;
    t(1, 0) = 4; t(1, 1) = 5; t(1, 2) = 6;
    Tensor tr = t.transpose();
    assert(tr(0, 0) == 1);
    assert(tr(1, 0) == 2);
    assert(tr(2, 0) == 3);
    assert(tr.rows == 3 && tr.cols == 2);
}

void test_xavier_init() {
    Tensor t(100, 100);
    t.xavier_init();
    float n = t.norm();
    assert(n > 0);
}

void test_eye() {
    Tensor e = Tensor::eye(3);
    assert(e(0, 0) == 1);
    assert(e(1, 1) == 1);
    assert(e(2, 2) == 1);
    assert(e(0, 1) == 0);
}

void test_row_normalize() {
    Tensor t(2, 2);
    t(0, 0) = 3; t(0, 1) = 4;
    t(1, 0) = 0; t(1, 1) = 0;
    Tensor n = t.row_normalize();
    assert(std::abs(n(0, 0) - 0.6f) < 1e-4);
    assert(std::abs(n(0, 1) - 0.8f) < 1e-4);
}

void test_argmax() {
    Tensor t(1, 5);
    t(0, 0) = 0.1f; t(0, 1) = 0.3f; t(0, 2) = 0.7f; t(0, 3) = 0.5f; t(0, 4) = 0.2f;
    assert(t.argmax() == 2);
}

void run_all_tests() {
    test_constructor_and_access();
    test_addition();
    test_multiplication();
    test_transpose();
    test_xavier_init();
    test_eye();
    test_row_normalize();
    test_argmax();
    std::cout << "✅ Todos los tests pasaron correctamente.\n";
}

int main() {
    run_all_tests();
    return 0;
}
