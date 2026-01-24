#include "../include/tensor.hpp"
#include <numeric>
#include <stdexcept>
#include <limits>

namespace hml::tensor {
    tensor::tensor() noexcept {
        shape_.clear();
        strides_.clear();
        data_.clear();
    }
    tensor::tensor(std::span<const size_t> dims){
        if (dims.empty()) {
            throw std::invalid_argument("tensor: shape must have at least 1 dimension");
        }    
        shape_.assign(dims.begin(), dims.end());

        std::size_t numel = 1;
        for (std::size_t d : shape_){
            if (d == 0){
                throw std::invalid_argument("tensor: dimensions must be > 0");
            }
            if (numel > std::numeric_limits<std::size_t>::max() / d) {
                throw std::overflow_error("tensor: numel overflow");
            }
            numel *= d;
        }
        strides_.resize(shape_.size());
        strides_.back() = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; i--){
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }

        data_.resize(numel);
    }

    tensor::tensor(std::initializer_list<size_t> dims) 
        : tensor(std::span<const std::size_t>(dims.begin(), dims.size())) {}

    tensor tensor::operator+(const tensor& x) const {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] += x.data_[i];
        }
        return res;
    }
    tensor tensor::operator+(float x) const {
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] += x;
        }
        return res;
    }

    tensor& tensor::operator+=(const tensor& x) {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] += x.data_[i];
        }
        return *this;
    }
    tensor& tensor::operator+=(float x) {
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] += x;
        }
        return *this;
    }

    tensor tensor::operator-(const tensor& x) const {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] -= x.data_[i];
        }
        return res;
    }
    tensor tensor::operator-(float x) const {
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] -= x;
        }
        return res;
    }

    tensor& tensor::operator-=(const tensor& x) {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] -= x.data_[i];
        }
        return *this;
    }
    tensor& tensor::operator-=(float x) {
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] -= x;
        }
        return *this;
    }

    tensor tensor::operator*(const tensor& x) const {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] *= x.data_[i];
        }
        return res;
    }
    tensor tensor::operator*(float x) const {
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] *= x;
        }
        return res;
    }

    tensor& tensor::operator*=(const tensor& x) {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] *= x.data_[i];
        }
        return *this;
    }
    tensor& tensor::operator*=(float x) {
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] *= x;
        }
        return *this;
    }

    tensor tensor::operator/(const tensor& x) const {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");

        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] /= x.data_[i];
        }
        return res;
    }

    tensor tensor::operator/(float x) const {
        tensor res = *this;
        for (std::size_t i = 0; i < data_.size(); i++){
            res.data_[i] /= x;
        }
        return res;
    }

    tensor& tensor::operator/=(const tensor& x) {
        if (this->shape_ != x.shape_) throw std::invalid_argument("tensor: tensors must be the same shape");
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] /= x.data_[i];
        }
        return *this;
    }
    tensor& tensor::operator/=(float x) {
        for (std::size_t i = 0; i < data_.size(); i++){
            this->data_[i] /= x;
        }
        return *this;
    }

    tensor tensor::transpose() const {
        if (this->shape_.size() < 2) throw std::invalid_argument("Can not transpose empty matrix");
        else if (this->shape_.size() == 2) {
            tensor res{this->shape_[1], this->shape_[0]};
            for (std::size_t i = 0; i < this->shape_[0]; i++) {
                for (std::size_t j = 0; j < this->shape_[1]; j++) {
                    res.data_[j * res.shape_[1] + i] = this->data_[i * this->shape_[1] + j];
                }   
            }
            return res;
        }
        else {
            std::vector<std::size_t> new_shape(this->shape_);
            std::swap(new_shape[new_shape.size() - 2], new_shape[new_shape.size() - 1]);
            tensor res{new_shape};
            std::size_t num_matrices = 1;
            for (std::size_t s = 0; s < res.shape_.size() - 2; s++){
                num_matrices *=  res.shape_[s];
            }
            std::size_t m = this->shape_[this->shape_.size() - 2];
            std::size_t n = this->shape_[this->shape_.size() - 1];
            for (std::size_t m_i = 0; m_i < num_matrices; m_i++){
                std::size_t offset = m_i * m * n;
                for (std::size_t i = 0; i < m; i++) {
                    for (std::size_t j = 0; j < n; j++){
                        res.data_[offset + j*m + i] = this->data_[offset + i * n + j];
                    }
                }
            } 
            return res;
        }
    }

    static std::size_t prod(const std::vector<std::size_t>& v, std::size_t start, std::size_t end){
        std::size_t out = 1;
        for (std::size_t i = start; i < end; i++) out *= v[i];
        return out;
    }

    tensor tensor::matmul(const tensor& x) const {
        if (!this->is_contiguous() || !x.is_contiguous())
            throw std::invalid_argument("tensor: matmul requires contiguous tensors");

        std::vector<std::size_t> a_shape = this->shape_;
        std::vector<std::size_t> b_shape = x.shape_;

        bool a_vec = false;
        bool b_vec = false;

        // Promote vectors to matrices
        if (a_shape.size() == 1) {
            a_vec = true;
            a_shape = {1, a_shape[0]};      // [n] -> [1, n]
        }
        if (b_shape.size() == 1) {
            b_vec = true;
            b_shape = {b_shape[0], 1};      // [n] -> [n, 1]
        }
    
        if (a_shape.size() < 2 || b_shape.size() < 2)
            throw std::invalid_argument("tensor: matmul requires tensors with ndim >= 1");

        const std::size_t a_m = a_shape[a_shape.size() - 2];
        const std::size_t a_n = a_shape[a_shape.size() - 1];
        const std::size_t b_n = b_shape[b_shape.size() - 2];
        const std::size_t b_p = b_shape[b_shape.size() - 1];

        if (a_n != b_n)
            throw std::invalid_argument("tensor: matmul requires [..., m, n] x [..., n, p]");

        const std::size_t a_rest = a_shape.size() - 2;
        const std::size_t b_rest = b_shape.size() - 2;

        std::vector<std::size_t> a_batch_shape(a_shape.begin(), a_shape.begin() + a_rest);
        std::vector<std::size_t> b_batch_shape(b_shape.begin(), b_shape.begin() + b_rest);

        // Simple batching rule (same as your intent):
        // - if both have batch dims, they must match exactly
        // - else output batch shape is the one that exists
        std::vector<std::size_t> out_batch_shape;
        if (!a_batch_shape.empty() && !b_batch_shape.empty()) {
            if (a_batch_shape != b_batch_shape)
                throw std::invalid_argument("tensor: batch dimensions must match (no broadcasting yet)");
            out_batch_shape = a_batch_shape;
        } else if (!a_batch_shape.empty()) {
            out_batch_shape = a_batch_shape;
        } else {
            out_batch_shape = b_batch_shape;
        }

        // Compute batch count
        auto prod_no_overflow = [&](const std::vector<std::size_t>& v) {
            std::size_t out = 1;
            for (std::size_t d : v) out *= d;
            return out;
        };
        const std::size_t batch_count = prod_no_overflow(out_batch_shape);

        // Build output shape
        std::vector<std::size_t> out_shape = out_batch_shape;
        if (a_vec && b_vec) {
            // dot product -> choose [1] or {} depending on your convention
            out_shape = {1};
        } else if (a_vec && !b_vec) {
            out_shape.push_back(b_p);       // [p]
        } else if (!a_vec && b_vec) {
            out_shape.push_back(a_m);       // [m]
        } else {
            out_shape.push_back(a_m);
            out_shape.push_back(b_p);
        }

        tensor out(std::span<const std::size_t>(out_shape.data(), out_shape.size()));

        const std::size_t A_block = a_m * a_n;
        const std::size_t B_block = a_n * b_p;

        // If one side has no batch dims, reuse its single matrix each batch
        const bool A_batched = !a_batch_shape.empty();
        const bool B_batched = !b_batch_shape.empty();

        const float* A = this->data_.data();
        const float* B = x.data_.data();
        float* C = out.data_.data();

        // For output, if you returned [m] or [p] for vector-ish results,
        // you still compute into an implicit [m,1] or [1,p] internally.
        const std::size_t C_m = a_m;
        const std::size_t C_p = b_p;
        const std::size_t C_block = C_m * C_p;

        for (std::size_t b = 0; b < batch_count; b++) {
            const std::size_t a_b = A_batched ? b : 0;
            const std::size_t b_b = B_batched ? b : 0;

            const std::size_t a_base = a_b * A_block;
            const std::size_t b_base = b_b * B_block;
            const std::size_t c_base = b * C_block;

            for (std::size_t i = 0; i < a_m; i++) {
                for (std::size_t j = 0; j < b_p; j++) {
                    float sum = 0.0f;
                    for (std::size_t t = 0; t < a_n; t++) {
                        sum += A[a_base + i * a_n + t] * B[b_base + t * b_p + j];
                    }
                    C[c_base + i * b_p + j] = sum;
                }
            }
        }

        return out;
    }

    tensor& reshape(std::span<const std::size_t> dims){
        std::size_t size = 1;
        for (std::size_t i = 0; i < dims.size(); i++) { size *= dims[i]; }
        if (size != this->size()) throw std::invalid_argument("Reshape must take same number of elements");
        this->shape_ = dims;
        this->strides_.back() = 1;
        for (int i = static_cast<int>(this->shape_.size()) - 2; i >= 0; i--){
            this->strides_[i] = this->strides_[i + 1] * this->shape_[i + 1];
        }
        return *this;
    }

   
    std::size_t tensor::ndim() const { return shape_.size(); } 
    std::size_t tensor::numel() const { return data_.size(); } 

    const std::vector<std::size_t>& tensor::get_shape() const noexcept { return shape_; } 
    const std::vector<float>& tensor::get_data() const noexcept { return data_; }
    
    const float* tensor::data() const noexcept { return data_.data(); }
    float* tensor::data() noexcept { return data_.data(); }
    std::size_t tensor::size() const noexcept { return data_.size(); }

    bool tensor::is_contiguous() const {
        if (this->shape_.empty()) { return true; }
        std::size_t expected = 1;
        for (std::size_t i = shape_.size(); i-- > 0; ){
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true; 
    } 

};
