// test_tensor.cpp
// Basic sanity tests for your tensor: constructors/strides/contiguity,
// elementwise ops, scalar ops, transpose (2D + batched), and batched matmul.
//
// Build example (adjust include/library paths as needed):
//   g++ -std=c++20 -O0 -g test_tensor.cpp ../src/tensor.cpp -I../include -o test_tensor
//   ./test_tensor

#include "../include/tensor.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>

using namespace std;
using hml::tensor::tensor;

static bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

static size_t prod(const vector<size_t>& v) {
    size_t out = 1;
    for (size_t x : v) out *= x;
    return out;
}

static void expect_shape(const tensor& t, const vector<size_t>& expected) {
    const auto& s = t.get_shape();
    if (s != expected) {
        cerr << "Shape mismatch!\nExpected: [";
        for (size_t i = 0; i < expected.size(); i++) cerr << expected[i] << (i+1<expected.size()? ", ":"");
        cerr << "]\nGot:      [";
        for (size_t i = 0; i < s.size(); i++) cerr << s[i] << (i+1<s.size()? ", ":"");
        cerr << "]\n";
        assert(false);
    }
}

static void fill_seq(tensor& t, float start = 0.0f, float step = 1.0f) {
    float* p = t.data();
    for (size_t i = 0; i < t.size(); i++) {
        p[i] = start + step * static_cast<float>(i);
    }
}

static void print_tensor_flat(const tensor& t, const string& name, size_t max_elems = 32) {
    cout << name << " shape=[";
    const auto& s = t.get_shape();
    for (size_t i = 0; i < s.size(); i++) cout << s[i] << (i+1<s.size()? ", ":"");
    cout << "], numel=" << t.size() << "\n";

    const float* p = t.data();
    cout << name << " data(flat): ";
    size_t n = min(max_elems, t.size());
    for (size_t i = 0; i < n; i++) {
        cout << p[i] << (i+1<n? ", ":"");
    }
    if (t.size() > n) cout << " ...";
    cout << "\n\n";
}

static void test_elementwise_ops() {
    cout << "=== test_elementwise_ops ===\n";

    tensor a{2, 3};
    tensor b{2, 3};
    fill_seq(a, 0.0f);      // 0..5
    fill_seq(b, 10.0f);     // 10..15

    tensor c = a + b;
    expect_shape(c, {2, 3});
    for (size_t i = 0; i < c.size(); i++) {
        assert(nearly_equal(c.data()[i], a.data()[i] + b.data()[i]));
    }

    tensor d = a * 2.0f;
    for (size_t i = 0; i < d.size(); i++) {
        assert(nearly_equal(d.data()[i], a.data()[i] * 2.0f));
    }

    tensor e = b - 3.0f;
    for (size_t i = 0; i < e.size(); i++) {
        assert(nearly_equal(e.data()[i], b.data()[i] - 3.0f));
    }

    tensor f = b / 2.0f;
    for (size_t i = 0; i < f.size(); i++) {
        assert(nearly_equal(f.data()[i], b.data()[i] / 2.0f));
    }

    // in-place
    tensor g{2, 3};
    fill_seq(g, 1.0f); // 1..6
    g += 5.0f;
    for (size_t i = 0; i < g.size(); i++) {
        assert(nearly_equal(g.data()[i], (1.0f + (float)i) + 5.0f));
    }

    cout << "OK\n\n";
}

static void test_transpose_2d() {
    cout << "=== test_transpose_2d ===\n";

    tensor a{2, 3};
    fill_seq(a, 0.0f); // row-major:
                       // [[0,1,2],
                       //  [3,4,5]]

    tensor at;
    try {
        at = a.transpose();
    } catch (const std::exception& e) {
        cerr << "transpose() threw: " << e.what() << "\n";
        assert(false);
    }

    expect_shape(at, {3, 2});

    // expected transpose:
    // [[0,3],
    //  [1,4],
    //  [2,5]]
    const float* p = at.data();
    assert(nearly_equal(p[0], 0.0f)); // (0,0)
    assert(nearly_equal(p[1], 3.0f)); // (0,1)
    assert(nearly_equal(p[2], 1.0f)); // (1,0)
    assert(nearly_equal(p[3], 4.0f)); // (1,1)
    assert(nearly_equal(p[4], 2.0f)); // (2,0)
    assert(nearly_equal(p[5], 5.0f)); // (2,1)

    cout << "OK\n\n";
}

static void test_transpose_batched_3d() {
    cout << "=== test_transpose_batched_3d ===\n";

    // shape: (B=2, M=2, N=3) -> transpose should give (2, 3, 2)
    tensor a{2, 2, 3};
    fill_seq(a, 0.0f);

    tensor at;
    try {
        at = a.transpose();
    } catch (const std::exception& e) {
        cerr << "batched transpose() threw: " << e.what() << "\n";
        assert(false);
    }

    expect_shape(at, {2, 3, 2});

    // Check by brute-force reference mapping:
    // For each batch b, matrix is 2x3; transpose to 3x2.
    const float* A = a.data();
    const float* T = at.data();

    size_t B = 2, M = 2, N = 3;
    size_t A_block = M * N;
    size_t T_block = N * M;

    for (size_t b = 0; b < B; b++) {
        size_t a_off = b * A_block;
        size_t t_off = b * T_block;
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float a_ij = A[a_off + i * N + j];
                float t_ji = T[t_off + j * M + i];
                if (!nearly_equal(a_ij, t_ji)) {
                    cerr << "Mismatch at batch " << b << " a("<<i<<","<<j<<") != t("<<j<<","<<i<<")\n";
                    cerr << "a=" << a_ij << " t=" << t_ji << "\n";
                    assert(false);
                }
            }
        }
    }

    cout << "OK\n\n";
}

static void test_batched_matmul() {
    cout << "=== test_batched_matmul ===\n";

    // A: (B=2, M=2, K=3)
    // B: (B=2, K=3, N=4)
    // C: (B=2, M=2, N=4)
    tensor A{2, 2, 3};
    tensor B{2, 3, 4};

    // deterministic small values
    fill_seq(A, 1.0f, 0.5f);
    fill_seq(B, -1.0f, 0.25f);

    tensor C;
    try {
        C = A.matmul(B);
    } catch (const std::exception& e) {
        cerr << "matmul() threw: " << e.what() << "\n";
        assert(false);
    }

    expect_shape(C, {2, 2, 4});

    // Reference compute
    const float* a = A.data();
    const float* b = B.data();
    const float* c = C.data();

    size_t batch = 2;
    size_t M = 2, K = 3, N = 4;
    size_t A_block = M * K;
    size_t B_block = K * N;
    size_t C_block = M * N;

    for (size_t bb = 0; bb < batch; bb++) {
        size_t a_off = bb * A_block;
        size_t b_off = bb * B_block;
        size_t c_off = bb * C_block;

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t t = 0; t < K; t++) {
                    sum += a[a_off + i*K + t] * b[b_off + t*N + j];
                }
                float got = c[c_off + i*N + j];
                if (!nearly_equal(sum, got, 1e-4f)) {
                    cerr << "Mismatch batch="<<bb<<" i="<<i<<" j="<<j
                         << " expected="<<sum<<" got="<<got<<"\n";
                    assert(false);
                }
            }
        }
    }

    cout << "OK\n\n";
}

static void test_vector_transpose_throws() {
    cout << "=== test_vector_transpose_throws ===\n";
    tensor v{5};
    fill_seq(v, 0.0f);

    bool threw = false;
    try {
        (void)v.transpose();
    } catch (...) {
        threw = true;
    }

    // Your current code says it throws when ndim < 2.
    // If you later change to no-op for 1D, update this test accordingly.
    assert(threw && "Expected transpose() to throw for 1D tensor in current implementation");

    cout << "OK\n\n";
}

int main() {
    try {
        test_elementwise_ops();
        test_transpose_2d();
        test_transpose_batched_3d();
        test_batched_matmul();
        test_vector_transpose_throws();

        cout << "ALL TESTS PASSED ✅\n";
    } catch (const std::exception& e) {
        cerr << "Unhandled exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

