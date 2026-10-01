#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#endif

namespace pokitlms::detail {

inline float dot_f32(const float* lhs, const float* rhs, std::size_t count) noexcept {
    std::size_t i = 0;
    float sum = 0.0F;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    float32x4_t accumulator = vdupq_n_f32(0.0F);
    for (; i + 4 <= count; i += 4) {
        accumulator = vfmaq_f32(accumulator, vld1q_f32(lhs + i), vld1q_f32(rhs + i));
    }
    sum = vaddvq_f32(accumulator);
#endif
    for (; i < count; ++i) sum += lhs[i] * rhs[i];
    return sum;
}

inline float dot_i8_f32(const float* lhs, const std::int8_t* quantized,
                        std::size_t count) noexcept {
    std::size_t i = 0;
    float sum = 0.0F;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    float32x4_t accumulator = vdupq_n_f32(0.0F);
    for (; i + 16 <= count; i += 16) {
        const int8x16_t q8 = vld1q_s8(quantized + i);
        const int16x8_t q16lo = vmovl_s8(vget_low_s8(q8));
        const int16x8_t q16hi = vmovl_s8(vget_high_s8(q8));
        const int32x4_t q32[4] = {
            vmovl_s16(vget_low_s16(q16lo)), vmovl_s16(vget_high_s16(q16lo)),
            vmovl_s16(vget_low_s16(q16hi)), vmovl_s16(vget_high_s16(q16hi))};
        for (std::size_t group = 0; group < 4; ++group) {
            accumulator = vfmaq_f32(accumulator,
                vcvtq_f32_s32(q32[group]), vld1q_f32(lhs + i + group * 4));
        }
    }
    sum = vaddvq_f32(accumulator);
#endif
    for (; i < count; ++i) sum += static_cast<float>(quantized[i]) * lhs[i];
    return sum;
}

inline void unpack_q4_0(const std::uint8_t* packed, std::int8_t* decoded) noexcept {
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    const auto values = vld1q_u8(packed);
    const auto low = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(values, vdupq_n_u8(0x0f))),
                              vdupq_n_s8(8));
    const auto high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(values, 4)),
                               vdupq_n_s8(8));
    vst1q_s8(decoded, low);
    vst1q_s8(decoded + 16, high);
#else
    for (std::size_t i = 0; i < 16; ++i) {
        decoded[i] = static_cast<std::int8_t>(static_cast<int>(packed[i] & 0x0fU) - 8);
        decoded[i + 16] = static_cast<std::int8_t>(static_cast<int>(packed[i] >> 4) - 8);
    }
#endif
}

inline void scale_i8_f32(float* output, const std::int8_t* values, float scale,
                         std::size_t count) noexcept {
    std::size_t i = 0;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    const float32x4_t factor = vdupq_n_f32(scale);
    for (; i + 16 <= count; i += 16) {
        const int8x16_t q8 = vld1q_s8(values + i);
        const int16x8_t q16lo = vmovl_s8(vget_low_s8(q8));
        const int16x8_t q16hi = vmovl_s8(vget_high_s8(q8));
        const int32x4_t q32[4] = {
            vmovl_s16(vget_low_s16(q16lo)), vmovl_s16(vget_high_s16(q16lo)),
            vmovl_s16(vget_low_s16(q16hi)), vmovl_s16(vget_high_s16(q16hi))};
        for (std::size_t group = 0; group < 4; ++group) {
            vst1q_f32(output + i + group * 4,
                vmulq_f32(vcvtq_f32_s32(q32[group]), factor));
        }
    }
#endif
    for (; i < count; ++i) output[i] = scale * static_cast<float>(values[i]);
}

inline void rope_rotate_f32(float* first, float* second, const float* cosine,
                            const float* sine, std::size_t count) noexcept {
    std::size_t i = 0;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    for (; i + 4 <= count; i += 4) {
        const auto a = vld1q_f32(first + i);
        const auto b = vld1q_f32(second + i);
        const auto c = vld1q_f32(cosine + i);
        const auto s = vld1q_f32(sine + i);
        vst1q_f32(first + i, vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, s)));
        vst1q_f32(second + i, vaddq_f32(vmulq_f32(b, c), vmulq_f32(a, s)));
    }
#endif
    for (; i < count; ++i) {
        const float a = first[i];
        const float b = second[i];
        first[i] = a * cosine[i] - b * sine[i];
        second[i] = b * cosine[i] + a * sine[i];
    }
}

inline void rms_scale_f32(const float* input, const float* weight, float* output,
                          float scale, std::size_t count) noexcept {
    std::size_t i = 0;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    const float32x4_t factor = vdupq_n_f32(scale);
    for (; i + 4 <= count; i += 4) {
        const auto values = vld1q_f32(input + i);
        const auto weights = vld1q_f32(weight + i);
        const auto scaled = vmulq_f32(values, factor);
        vst1q_f32(output + i, vmulq_f32(scaled, weights));
    }
#endif
    for (; i < count; ++i) output[i] = input[i] * scale * weight[i];
}

inline void scale_add_f32(float* destination, const float* source, float scale,
                          std::size_t count) noexcept {
    std::size_t i = 0;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    const float32x4_t factor = vdupq_n_f32(scale);
    for (; i + 4 <= count; i += 4) {
        const auto dst = vld1q_f32(destination + i);
        const auto src = vld1q_f32(source + i);
        vst1q_f32(destination + i, vfmaq_f32(dst, src, factor));
    }
#endif
    for (; i < count; ++i) destination[i] += scale * source[i];
}

inline void scale_add_i8_f32(float* destination, const std::int8_t* source,
                             float scale, std::size_t count) noexcept {
    std::size_t i = 0;
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    const float32x4_t factor = vdupq_n_f32(scale);
    for (; i + 16 <= count; i += 16) {
        const int8x16_t q8 = vld1q_s8(source + i);
        const int16x8_t q16lo = vmovl_s8(vget_low_s8(q8));
        const int16x8_t q16hi = vmovl_s8(vget_high_s8(q8));
        const int32x4_t q32[4] = {
            vmovl_s16(vget_low_s16(q16lo)), vmovl_s16(vget_high_s16(q16lo)),
            vmovl_s16(vget_low_s16(q16hi)), vmovl_s16(vget_high_s16(q16hi))};
        for (std::size_t group = 0; group < 4; ++group) {
            const auto dst = vld1q_f32(destination + i + group * 4);
            const auto values = vcvtq_f32_s32(q32[group]);
            vst1q_f32(destination + i + group * 4, vfmaq_f32(dst, values, factor));
        }
    }
#endif
    for (; i < count; ++i) destination[i] += scale * static_cast<float>(source[i]);
}

}  // namespace pokitlms::detail
