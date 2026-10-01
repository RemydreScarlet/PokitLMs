#include "pokitlms/tensor.hpp"

#include <numeric>
#include <stdexcept>

namespace pokitlms {

namespace {

std::size_t compute_numel(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    std::size_t result = 1;
    for (const auto dim : shape) {
        if (dim == 0) {
            return 0;
        }
        if (result > static_cast<std::size_t>(-1) / dim) {
            throw std::overflow_error("tensor size overflow");
        }
        result *= dim;
    }
    return result;
}

}  // namespace

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)),
      storage_(compute_numel(shape_)) {}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::numel() const noexcept {
    return storage_.size();
}

DType Tensor::dtype() const noexcept {
    return DType::F32;
}

std::span<float> Tensor::data() noexcept {
    return storage_;
}

std::span<const float> Tensor::data() const noexcept {
    return storage_;
}

}  // namespace pokitlms
