#pragma once
#include <vector>
#include <tuple>
#include <initializer_list>
#include <cstddef>
#include <span>

namespace hml::tensor {
    class tensor {
        public:
            tensor() noexcept;
            explicit tensor(std::span<const std::size_t> dims);
            tensor(std::initializer_list<std::size_t> args);

            tensor operator+(const tensor& x) const;
            tensor operator+(float x) const;
            
            tensor& operator+=(const tensor& x);
            tensor& operator+=(float x);
            
            tensor operator-(const tensor& x) const;
            tensor operator-(float x) const;

            tensor& operator-=(const tensor& x);
            tensor& operator-=(float x);
            
            tensor operator*(const tensor& x) const;
            tensor operator*(float x) const;

            tensor& operator*=(const tensor& x);
            tensor& operator*=(float x);

            tensor operator/(const tensor& x) const;
            tensor operator/(float x) const;

            tensor& operator/=(const tensor& x);
            tensor& operator/=(float x);

            tensor transpose() const;	

            tensor matmul(const tensor& x) const;
            
            std::size_t ndim() const;
            std::size_t numel() const;
            const std::vector<std::size_t>& get_shape() const noexcept;
            const std::vector<float>& get_data() const noexcept;
            bool is_contiguous() const;

            const float* data() const noexcept;
            float* data() noexcept;
            std::size_t size() const noexcept; 

            tensor& reshape(std::span<const std::size_t> dims);
            tensor unsqueeze(std::size_t axis);
            tensor squeeze(std::size_t axis);
            tensor& permute(std::span<const std::size_t> axes);

        private:
                std::vector<float> data_;
                std::vector<std::size_t> shape_;
                std::vector<std::size_t> strides_;

    };
}
