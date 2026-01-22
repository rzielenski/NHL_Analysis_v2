#include "../include/tensor.hpp"
#include <numeric>
#include <stdexcept>
#include <limits>

namespace hml::tensor {
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

    tensor tensor::matmul(const tensor& x) const;


    size_t tensor::ndim() const { return shape_.size(); } 
    size_t tensor::numel() const { return data_.size(); } 

    const std::vector<size_t>& tensor::get_shape() const { return shape_; } 
    const std::vector<float>& tensor::get_data() const { return data_; }
    
    bool tensor::is_contiguous() const {
        if (this->shape_.empty()) { return true; }
        std::size_t expected = 1;
        for (std::size_t i = shape_.size(); i-- > 0; ){
            if (strides_[i] != expected) return false;
            expected *= shape_[i];
        }
        return true; 
    } 

}
