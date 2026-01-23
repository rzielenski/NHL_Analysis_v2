#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "../include/tensor.hpp"

using hml::tensor::tensor;

static bool feq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static void expect_true(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error("FAIL: " + msg);
}

static void expect_eq_size(std::size_t a, std::size_t b, const std::string& msg) {
    if (a != b) {
        throw std::runtime_error("FAIL: " + msg + " (got " + std::to_string(a) +
                                 ", expected " + std::to_string(b) + ")");
    }
}

static void expect_shape(const tensor& t, const std::vector<std::size_t>& shape, const std::string& msg) {
    const auto& s = t.get_shape();
    expect_eq_size(s.size(), shape.size(), msg + " shape.ndim mismatch");
    for (std::size_t i = 0; i < s.size(); i++) {
        if (s[i] != shape[i]) {
            throw std::runtime_error("FAIL: " + msg + " shape mismatch at dim " + std::to_string(i) +
                                     " (got " + std::to_string(s[i]) + ", expected " + std::to_string(shape[i]) + ")");
        }
    }
}

static void expect_data(const tensor& t, const std::vector<float>& expected, const std::string& msg) {
    const auto& d = t.get_data();
    expect_eq_size(d.size(), expected.size(), msg + " data.size mismatch");
    for (std::size_t i = 0; i < d.size(); i++) {
        if (!feq(d[i], expected[i])) {
            throw std::runtime_error("FAIL: " + msg + " data mismatch at idx " + std::to_string(i) +
                                     " (got " + std::to_string(d[i]) + ", expected " + std::to_string(expected[i]) + ")");
        }
    }
}

static void fill_seq(tensor& t, float start = 0.0f, float step = 1.0f) {
    float* p = t.data();
    float v = start;
    for (std::size_t i = 0; i < t.size(); i++) {
        p[i] = v;
        v += step;
    }
}

static void set_data(tensor& t, std::initializer_list<float> vals) {
    expect_eq_size(t.size(), vals.size(), "set_data size mismatch");
    float* p = t.data();
    std::size_t i = 0;
    for (float v : vals) {
        p[i++] = v;
    }
}

int main() {
    try {
        std::cout << "Running tensor tests...\n";

        // ctor + shape/numel/contiguous
        {
            tensor a({2, 3, 4});
            expect_shape(a, {2, 3, 4}, "ctor shape");
            expect_eq_size(a.ndim(), 3, "ndim");
            expect_eq_size(a.numel(), 24, "numel");
            expect_true(a.is_contiguous(), "is_contiguous true");
        }

        // elementwise scalar ops
        {
            tensor a({2, 2});
            fill_seq(a, 1.0f); // [1,2,3,4]
            tensor b = a + 2.0f; // [3,4,5,6]
            expect_data(b, {3,4,5,6}, "a + scalar");

            b -= 1.0f; // [2,3,4,5]
            expect_data(b, {2,3,4,5}, "b -= scalar");

            tensor c = b * 2.0f; // [4,6,8,10]
            expect_data(c, {4,6,8,10}, "b * scalar");

            tensor d = c / 2.0f; // [2,3,4,5]
            expect_data(d, {2,3,4,5}, "c / scalar");
        }

        // elementwise tensor ops
        {
            tensor a({2, 3});
            tensor b({2, 3});
            fill_seq(a, 1.0f);  // 1..6
            fill_seq(b, 10.0f); // 10..15

            tensor c = a + b; // 11,13,15,17,19,21
            expect_data(c, {11,13,15,17,19,21}, "a + b");

            c -= a; // back to 10..15
            expect_data(c, {10,11,12,13,14,15}, "c -= a");

            tensor e = b - a; // all 9
            expect_data(e, {9,9,9,9,9,9}, "b - a");

            tensor f = a * a; // squares
            expect_data(f, {1,4,9,16,25,36}, "a * a");
        }

        // matmul: 2D @ 2D
        // A 2x3, B 3x2 => C 2x2
        // C = [58 64
        //      139 154]
        {
            tensor A({2, 3});
            tensor B({3, 2});
            set_data(A, {1,2,3,4,5,6});
            set_data(B, {7,8,9,10,11,12});

            tensor C = A.matmul(B);
            expect_shape(C, {2, 2}, "matmul 2D@2D shape");
            expect_data(C, {58,64,139,154}, "matmul 2D@2D data");
        }

        // matmul: 1D @ 2D
        // [1 2 3] @ B => [58 64]
        {
            tensor v({3});
            tensor B({3, 2});
            set_data(v, {1,2,3});
            set_data(B, {7,8,9,10,11,12});

            tensor out = v.matmul(B);
            expect_shape(out, {2}, "matmul 1D@2D shape");
            expect_data(out, {58,64}, "matmul 1D@2D data");
        }

        // matmul: 2D @ 1D
        // A @ [1 2 3]^T => [14, 32]
        {
            tensor A({2, 3});
            tensor v({3});
            set_data(A, {1,2,3,4,5,6});
            set_data(v, {1,2,3});

            tensor out = A.matmul(v);
            expect_shape(out, {2}, "matmul 2D@1D shape");
            expect_data(out, {14,32}, "matmul 2D@1D data");
        }

        // matmul: 1D @ 1D (dot)
        // dot([1 2 3], [4 5 6]) = 32
        {
            tensor a({3});
            tensor b({3});
            set_data(a, {1,2,3});
            set_data(b, {4,5,6});

            tensor out = a.matmul(b);
            // if your convention differs (scalar vs {1}), adjust this:
            expect_shape(out, {1}, "matmul 1D@1D shape");
            expect_data(out, {32}, "matmul 1D@1D data");
        }

        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

