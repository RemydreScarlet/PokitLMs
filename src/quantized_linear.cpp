#include "pokitlms/ops/quantized_linear.hpp"
#include "simd_kernels.hpp"

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

void decode_q4_1(const std::byte* block, std::array<float, 32>& values) {
    const float d = half_to_float(read_u16(block));
    const float minimum = half_to_float(read_u16(block + 2));
    const auto* quant = block + 4;
    for (std::size_t i = 0; i < 16; ++i) {
        const auto packed = std::to_integer<std::uint8_t>(quant[i]);
        values[i] = static_cast<float>(packed & 0x0fU) * d + minimum;
        values[i + 16] = static_cast<float>(packed >> 4) * d + minimum;
    }
}

void decode_q5_0(const std::byte* block, std::array<float, 32>& values) {
    const float d = half_to_float(read_u16(block));
    const auto high_bits = read_u32(block + 2);
    const auto* quant = block + 6;
    for (std::size_t i = 0; i < 16; ++i) {
        const auto packed = std::to_integer<std::uint8_t>(quant[i]);
        const auto low = static_cast<int>(packed & 0x0fU);
        const auto high = static_cast<int>(packed >> 4);
        const auto q0 = low | static_cast<int>(((high_bits >> i) & 1U) << 4);
        const auto q1 = high | static_cast<int>(((high_bits >> (i + 16)) & 1U) << 4);
        values[i] = static_cast<float>(q0 - 16) * d;
        values[i + 16] = static_cast<float>(q1 - 16) * d;
    }
}

void decode_q5_1(const std::byte* block, std::array<float, 32>& values) {
    const float d = half_to_float(read_u16(block));
    const float minimum = half_to_float(read_u16(block + 2));
    const auto high_bits = read_u32(block + 4);
    const auto* quant = block + 8;
    for (std::size_t i = 0; i < 16; ++i) {
        const auto packed = std::to_integer<std::uint8_t>(quant[i]);
        const auto low = static_cast<unsigned>(packed & 0x0fU) |
            static_cast<unsigned>((high_bits >> i) & 1U) << 4;
        const auto high = static_cast<unsigned>(packed >> 4) |
            static_cast<unsigned>((high_bits >> (i + 16)) & 1U) << 4;
        values[i] = static_cast<float>(low) * d + minimum;
        values[i + 16] = static_cast<float>(high) * d + minimum;
    }
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

void decode_q5_k(const std::byte* block, std::array<float, 256>& values) {
    const float d = half_to_float(read_u16(block));
    const float dmin = half_to_float(read_u16(block + 2));
    const auto* packed_scales = block + 4;
    const auto* quants = block + 16;
    const auto* high = block + 144;
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
    std::size_t out = 0;
    std::size_t is = 0;
    std::uint8_t mask1 = 1, mask2 = 2;
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
        const auto [s0, m0] = scale_min(is++);
        const auto [s1, m1] = scale_min(is++);
        const float ds0 = d * s0, dm0 = dmin * m0;
        const float ds1 = d * s1, dm1 = dmin * m1;
        for (std::size_t i = 0; i < 32; ++i) {
            const auto qh = std::to_integer<std::uint8_t>(high[i]);
            const auto ql = std::to_integer<std::uint8_t>(quants[i]);
            values[out++] = ds0 * ((ql & 0x0fU) + ((qh & mask1) ? 16U : 0U)) - dm0;
        }
        for (std::size_t i = 0; i < 32; ++i) {
            const auto qh = std::to_integer<std::uint8_t>(high[i]);
            const auto ql = std::to_integer<std::uint8_t>(quants[i]);
            values[out++] = ds1 * ((ql >> 4) + ((qh & mask2) ? 16U : 0U)) - dm1;
        }
        quants += 32;
        mask1 = static_cast<std::uint8_t>(mask1 << 2);
        mask2 = static_cast<std::uint8_t>(mask2 << 2);
        if (mask1 == 0) { high += 32; mask1 = 1; mask2 = 2; }
    }
}

void decode_q6_k(const std::byte* block, std::array<float, 256>& values) {
    const auto* low = block;
    const auto* high = block + 128;
    const auto* scales = block + 192;
    const float d = half_to_float(read_u16(block + 208));
    for (std::size_t half = 0; half < 2; ++half) {
        const auto base = half * 128;
        for (std::size_t lane = 0; lane < 32; ++lane) {
            const auto is = lane / 16;
            const auto lo0 = std::to_integer<std::uint8_t>(low[lane]);
            const auto lo1 = std::to_integer<std::uint8_t>(low[lane + 32]);
            const auto hi = std::to_integer<std::uint8_t>(high[lane]);
            const int q0 = static_cast<int>((lo0 & 0x0fU) | ((hi & 0x03U) << 4)) - 32;
            const int q1 = static_cast<int>((lo1 & 0x0fU) | ((hi >> 2 & 0x03U) << 4)) - 32;
            const int q2 = static_cast<int>((lo0 >> 4) | ((hi >> 4 & 0x03U) << 4)) - 32;
            const int q3 = static_cast<int>((lo1 >> 4) | ((hi >> 6 & 0x03U) << 4)) - 32;
            const auto s0 = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(scales[half * 8 + is]));
            const auto s1 = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(scales[half * 8 + is + 2]));
            const auto s2 = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(scales[half * 8 + is + 4]));
            const auto s3 = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(scales[half * 8 + is + 6]));
            values[base + lane] = d * s0 * q0;
            values[base + lane + 32] = d * s1 * q1;
            values[base + lane + 64] = d * s2 * q2;
            values[base + lane + 96] = d * s3 * q3;
        }
        low += 64;
        high += 32;
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
            sum += detail::dot_f32(decoded.data(), input.data() + offset, BlockElements);
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
                std::array<std::int8_t, kBlockElements> decoded{};
                for (std::size_t i = 0; i < 16; ++i) {
                    const auto packed = std::to_integer<std::uint8_t>(quant[i]);
                    decoded[i] = static_cast<std::int8_t>(static_cast<int>(packed & 0x0fU) - 8);
                    decoded[i + 16] = static_cast<std::int8_t>(static_cast<int>(packed >> 4) - 8);
                }
                sum += scale * detail::dot_i8_f32(input.data() + input_offset, decoded.data(), kBlockElements);
            } else {
                std::array<std::int8_t, kBlockElements> decoded{};
                for (std::size_t i = 0; i < kBlockElements; ++i) {
                    decoded[i] = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(quant[i]));
                }
                sum += scale * detail::dot_i8_f32(input.data() + input_offset, decoded.data(), kBlockElements);
            }
        }
        output[row] = sum;
    }
}

}  // namespace

void dequantize_quantized_row(std::uint32_t type,
                              std::span<const std::byte> encoded,
                              std::span<float> output) {
    if (type == 2 || type == 8) {
        constexpr std::size_t block_elements = 32;
        const std::size_t block_bytes = type == 2 ? 18 : 34;
        if (output.empty() || output.size() % block_elements != 0 ||
            encoded.size() != output.size() / block_elements * block_bytes) {
            throw std::invalid_argument("invalid Q4_0/Q8_0 row dimensions");
        }
        for (std::size_t block = 0; block < output.size() / block_elements; ++block) {
            const auto* data = encoded.data() + block * block_bytes;
            const float scale = half_to_float(read_u16(data));
            if (type == 2) {
                for (std::size_t i = 0; i < 16; ++i) {
                    const auto packed = std::to_integer<std::uint8_t>(data[2 + i]);
                    output[block * 32 + i] = scale * (static_cast<int>(packed & 0x0fU) - 8);
                    output[block * 32 + i + 16] = scale * (static_cast<int>(packed >> 4) - 8);
                }
            } else {
                for (std::size_t i = 0; i < 32; ++i) {
                    const auto q = static_cast<std::int8_t>(std::to_integer<std::uint8_t>(data[2 + i]));
                    output[block * 32 + i] = scale * static_cast<float>(q);
                }
            }
        }
        return;
    }
    if (type == 3 || type == 6 || type == 7) {
        constexpr std::size_t block_elements = 32;
        const std::size_t block_bytes = type == 3 ? 20 : type == 6 ? 22 : 24;
        if (output.empty() || output.size() % block_elements != 0 ||
            encoded.size() != output.size() / block_elements * block_bytes) {
            throw std::invalid_argument("invalid Q4_1/Q5_0/Q5_1 row dimensions");
        }
        std::array<float, block_elements> decoded{};
        for (std::size_t block = 0; block < output.size() / block_elements; ++block) {
            const auto* data = encoded.data() + block * block_bytes;
            if (type == 3) decode_q4_1(data, decoded);
            else if (type == 6) decode_q5_0(data, decoded);
            else decode_q5_1(data, decoded);
            std::copy(decoded.begin(), decoded.end(), output.begin() +
                      static_cast<std::ptrdiff_t>(block * block_elements));
        }
        return;
    }
    if (type == 10 || type == 11 || type == 12 || type == 13 || type == 14) {
        constexpr std::size_t block_elements = 256;
        const std::size_t block_bytes = type == 10 ? 84 : type == 11 ? 110 :
            type == 12 ? 144 : type == 13 ? 176 : 210;
        if (output.empty() || output.size() % block_elements != 0 ||
            encoded.size() != output.size() / block_elements * block_bytes) {
            throw std::invalid_argument("invalid K-quantized row dimensions");
        }
        std::array<float, block_elements> decoded{};
        for (std::size_t block = 0; block < output.size() / block_elements; ++block) {
            const auto* data = encoded.data() + block * block_bytes;
            if (type == 10) decode_q2_k(data, decoded);
            else if (type == 11) decode_q3_k(data, decoded);
            else if (type == 12) decode_q4_k(data, decoded);
            else if (type == 13) decode_q5_k(data, decoded);
            else decode_q6_k(data, decoded);
            std::copy(decoded.begin(), decoded.end(), output.begin() +
                      static_cast<std::ptrdiff_t>(block * block_elements));
        }
        return;
    }
    throw std::invalid_argument("unsupported GGUF quantized type for row decoding");
}

void linear_q4_0(std::span<const float> input,
                 std::span<const std::byte> weights,
                 std::size_t output_features,
                 std::span<const float> bias,
                 std::span<float> output) {
    linear_quantized<true>(input, weights, output_features, bias, output);
}

void linear_q4_1(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<32, 20>(input, weights, output_features, bias, output, decode_q4_1);
}

void linear_q5_0(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<32, 22>(input, weights, output_features, bias, output, decode_q5_0);
}

void linear_q5_1(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<32, 24>(input, weights, output_features, bias, output, decode_q5_1);
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

void linear_q6_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<256, 210>(input, weights, output_features, bias, output, decode_q6_k);
}

void linear_q5_k(std::span<const float> input, std::span<const std::byte> weights,
                 std::size_t output_features, std::span<const float> bias,
                 std::span<float> output) {
    linear_k<256, 176>(input, weights, output_features, bias, output, decode_q5_k);
}

}  // namespace pokitlms
