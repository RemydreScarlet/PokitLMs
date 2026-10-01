#include "pokitlms/ops/quantized_linear.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pokitlms {
namespace {

constexpr std::size_t kBlockElements = 32;

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fU;
    std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) return std::bit_cast<float>(sign);
        int unbiased_exponent = -14;
        while ((mantissa & 0x0400U) == 0) {
            mantissa <<= 1;
            --unbiased_exponent;
        }
        mantissa &= 0x03ffU;
        result = sign | (static_cast<std::uint32_t>(unbiased_exponent + 127) << 23) |
                 (mantissa << 13);
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(result);
}

std::uint16_t read_u16(const std::byte* bytes) {
    return static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[0])) |
           static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1]) << 8);
}

template <bool Q4>
void linear_quantized(std::span<const float> input,
                      std::span<const std::byte> weights,
                      std::size_t output_features,
                      std::span<const float> bias,
                      std::span<float> output) {
    constexpr std::size_t block_bytes = Q4 ? 18 : 34;
    if (input.empty() || input.size() % kBlockElements != 0 || output_features == 0 ||
        output.size() != output_features || (!bias.empty() && bias.size() != output_features)) {
        throw std::invalid_argument("invalid quantized linear dimensions");
    }
    const auto blocks_per_row = input.size() / kBlockElements;
    if (output_features > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        output_features * blocks_per_row > std::numeric_limits<std::size_t>::max() / block_bytes ||
        weights.size() != output_features * blocks_per_row * block_bytes) {
        throw std::invalid_argument("quantized weight byte size does not match dimensions");
    }

    for (std::size_t row = 0; row < output_features; ++row) {
        float sum = bias.empty() ? 0.0F : bias[row];
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto* encoded = weights.data() + (row * blocks_per_row + block) * block_bytes;
            const float scale = half_to_float(read_u16(encoded));
            const auto* quant = encoded + 2;
            const auto input_offset = block * kBlockElements;
            if constexpr (Q4) {
                for (std::size_t i = 0; i < 16; ++i) {
                    const auto packed = std::to_integer<std::uint8_t>(quant[i]);
                    const int q0 = static_cast<int>(packed & 0x0fU) - 8;
                    const int q1 = static_cast<int>(packed >> 4) - 8;
                    sum += scale * static_cast<float>(q0) * input[input_offset + i];
                    sum += scale * static_cast<float>(q1) * input[input_offset + i + 16];
                }
            } else {
                for (std::size_t i = 0; i < kBlockElements; ++i) {
                    const auto q = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(quant[i]));
                    sum += scale * static_cast<float>(q) * input[input_offset + i];
                }
            }
        }
        output[row] = sum;
    }
}

}  // namespace

void linear_q4_0(std::span<const float> input,
                 std::span<const std::byte> weights,
                 std::size_t output_features,
                 std::span<const float> bias,
                 std::span<float> output) {
    linear_quantized<true>(input, weights, output_features, bias, output);
}

void linear_q8_0(std::span<const float> input,
                 std::span<const std::byte> weights,
                 std::size_t output_features,
                 std::span<const float> bias,
                 std::span<float> output) {
    linear_quantized<false>(input, weights, output_features, bias, output);
}

}  // namespace pokitlms
