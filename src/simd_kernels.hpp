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
