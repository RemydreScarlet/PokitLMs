#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pokitlms {

enum class DType : std::uint8_t {
    F32,
};

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] std::size_t numel() const noexcept;
    [[nodiscard]] DType dtype() const noexcept;

    [[nodiscard]] std::span<float> data() noexcept;
    [[nodiscard]] std::span<const float> data() const noexcept;

private:
    std::vector<std::size_t> shape_;
    std::vector<float> storage_;
};

}  // namespace pokitlms
