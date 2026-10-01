#include "pokitlms/ops/quantized_linear.hpp"

#include <bit>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

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

std::uint32_t read_u32(const std::byte* bytes) {
    return static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[0])) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[1])) << 8) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[2])) << 16) |
           (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(bytes[3])) << 24);
}

void decode_q2_k(const std::byte* block, std::array<float, 256>& values) {
    const auto* scales = block;
    const auto* quants = block + 16;
    const float d = half_to_float(read_u16(block + 80));
    const float dmin = half_to_float(read_u16(block + 82));
    std::size_t si = 0;
    std::size_t out = 0;
    for (std::size_t chunk = 0; chunk < 2; ++chunk) {
        unsigned shift = 0;
        for (std::size_t group = 0; group < 4; ++group) {
            const auto sc0 = std::to_integer<std::uint8_t>(scales[si++]);
            const auto sc1 = std::to_integer<std::uint8_t>(scales[si++]);
            const float dl0 = d * static_cast<float>(sc0 & 0x0fU);
            const float ml0 = dmin * static_cast<float>(sc0 >> 4);
            const float dl1 = d * static_cast<float>(sc1 & 0x0fU);
            const float ml1 = dmin * static_cast<float>(sc1 >> 4);
            for (std::size_t i = 0; i < 16; ++i) {
                values[out++] = dl0 * static_cast<float>((std::to_integer<std::uint8_t>(quants[i]) >> shift) & 3U) - ml0;
            }
            for (std::size_t i = 0; i < 16; ++i) {
                values[out++] = dl1 * static_cast<float>((std::to_integer<std::uint8_t>(quants[i + 16]) >> shift) & 3U) - ml1;
            }
            shift += 2;
        }
        quants += 32;
    }
}

void decode_q3_k(const std::byte* block, std::array<float, 256>& values) {
    const auto* quants = block + 2;
    const auto* high_mask = block + 66;
    const auto* packed_scales = block + 98;
    std::array<std::uint32_t, 4> aux{};
    aux[0] = read_u32(packed_scales);
    aux[1] = read_u32(packed_scales + 4);
    aux[2] = read_u32(packed_scales + 8);
    const auto tmp = aux[2];
    constexpr std::uint32_t mask1 = 0x03030303U;
    constexpr std::uint32_t mask2 = 0x0f0f0f0fU;
    aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4);
    aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4);
    aux[0] = (aux[0] & mask2) | (((tmp >> 0) & mask1) << 4);
    aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4);
    std::array<std::int8_t, 16> scales{};
    std::memcpy(scales.data(), aux.data(), 16);
    const float d = half_to_float(read_u16(block));
    std::size_t out = 0;
    std::size_t si = 0;
    std::uint8_t bit = 1;
    for (std::size_t chunk = 0; chunk < 2; ++chunk) {
        unsigned shift = 0;
        for (std::size_t group = 0; group < 4; ++group) {
            const float ds0 = d * static_cast<float>(scales[si++] - 32);
            const float ds1 = d * static_cast<float>(scales[si++] - 32);
            for (std::size_t i = 0; i < 16; ++i) {
                const auto q = (std::to_integer<std::uint8_t>(quants[i]) >> shift) & 3U;
                const auto hm = std::to_integer<std::uint8_t>(high_mask[i]);
                values[out++] = ds0 * (static_cast<int>(q) - ((hm & bit) ? 0 : 4));
            }
            for (std::size_t i = 0; i < 16; ++i) {
                const auto q = (std::to_integer<std::uint8_t>(quants[i + 16]) >> shift) & 3U;
                const auto hm = std::to_integer<std::uint8_t>(high_mask[i + 16]);
                values[out++] = ds1 * (static_cast<int>(q) - ((hm & bit) ? 0 : 4));
            }
            shift += 2;
            bit = static_cast<std::uint8_t>(bit << 1);
        }
        quants += 32;
    }
}

void decode_q4_k(const std::byte* block, std::array<float, 256>& values) {
    const float d = half_to_float(read_u16(block));
    const float dmin = half_to_float(read_u16(block + 2));
    const auto* packed_scales = block + 4;
    const auto* quants = block + 16;
    std::size_t out = 0;
    std::size_t scale_index = 0;
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
        const auto j0 = scale_index++;
        const auto j1 = scale_index++;
        auto scale_min = [packed_scales](std::size_t j) {
            std::pair<std::uint8_t, std::uint8_t> result;
            if (j < 4) {
                result.first = std::to_integer<std::uint8_t>(packed_scales[j]) & 63U;
                result.second = std::to_integer<std::uint8_t>(packed_scales[j + 4]) & 63U;
            } else {
                result.first = (std::to_integer<std::uint8_t>(packed_scales[j + 4]) & 0x0fU) |
                    ((std::to_integer<std::uint8_t>(packed_scales[j - 4]) >> 6) << 4);
                result.second = (std::to_integer<std::uint8_t>(packed_scales[j + 4]) >> 4) |
                    ((std::to_integer<std::uint8_t>(packed_scales[j]) >> 6) << 4);
            }
            return result;
        };
        const auto [s0, m0] = scale_min(j0);
        const auto [s1, m1] = scale_min(j1);
        const float ds0 = d * s0;
        const float dm0 = dmin * m0;
        const float ds1 = d * s1;
        const float dm1 = dmin * m1;
        for (std::size_t i = 0; i < 32; ++i) {
            const auto packed = std::to_integer<std::uint8_t>(quants[i]);
            values[out++] = ds0 * (packed & 0x0fU) - dm0;
        }
        for (std::size_t i = 0; i < 32; ++i) {
            const auto packed = std::to_integer<std::uint8_t>(quants[i]);
            values[out++] = ds1 * (packed >> 4) - dm1;
        }
        quants += 32;
    }
}

template <std::size_t BlockElements, std::size_t BlockBytes, typename Decoder>
void linear_k(std::span<const float> input, std::span<const std::byte> weights,
              std::size_t output_features, std::span<const float> bias,
              std::span<float> output, Decoder decode) {
    if (input.empty() || input.size() % BlockElements != 0 || output_features == 0 ||
        output.size() != output_features || (!bias.empty() && bias.size() != output_features)) {
        throw std::invalid_argument("invalid K-quantized linear dimensions");
    }
    const auto blocks_per_row = input.size() / BlockElements;
    if (output_features > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
        output_features * blocks_per_row > std::numeric_limits<std::size_t>::max() / BlockBytes ||
        weights.size() != output_features * blocks_per_row * BlockBytes) {
        throw std::invalid_argument("K-quantized weight byte size does not match dimensions");
    }
    std::array<float, BlockElements> decoded{};
    for (std::size_t row = 0; row < output_features; ++row) {
        float sum = bias.empty() ? 0.0F : bias[row];
        for (std::size_t block = 0; block < blocks_per_row; ++block) {
            const auto* encoded = weights.data() + (row * blocks_per_row + block) * BlockBytes;
            decode(encoded, decoded);
            const auto offset = block * BlockElements;
            for (std::size_t i = 0; i < BlockElements; ++i) sum += decoded[i] * input[offset + i];
        }
        output[row] = sum;
    }
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

void linear_q2_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<256, 84>(input, weights, output_features, bias, output, decode_q2_k);
}

void linear_q3_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<256, 110>(input, weights, output_features, bias, output, decode_q3_k);
}

void linear_q4_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<256, 144>(input, weights, output_features, bias, output, decode_q4_k);
}

}  // namespace pokitlms
