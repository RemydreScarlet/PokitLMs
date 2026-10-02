#include "pokitlms/kv_cache.hpp"
#include "pokitlms/ops/attention.hpp"
#include "pokitlms/ops/gated_delta_net.hpp"
#include "pokitlms/ops/linear.hpp"
#include "pokitlms/ops/rms_norm.hpp"
#include "pokitlms/ops/rope.hpp"
#include "pokitlms/pokitlms.h"
#include "pokitlms/tensor.hpp"

#include <cassert>
#include <cmath>
#include <string>
#include <string_view>

namespace {

bool nearly_equal(float a, float b, float tolerance = 1.0e-5F) {
    return std::fabs(a - b) <= tolerance;
}

void test_tensor_and_version() {
    pokitlms::Tensor tensor({4});
    assert(tensor.numel() == 4);
    const auto expected_version = std::to_string(POKITLMS_VERSION_MAJOR) + "." +
        std::to_string(POKITLMS_VERSION_MINOR) + "." + std::to_string(POKITLMS_VERSION_PATCH);
    assert(std::string_view(pokitlms_version()) == expected_version);
}

void test_rms_norm() {
    const float input[] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {};

    pokitlms::rms_norm(input, weight, output);

    const float mean_square = (1.0F + 4.0F + 9.0F + 16.0F) / 4.0F;
    const float expected0 = 1.0F / std::sqrt(mean_square + 1.0e-5F);
    assert(nearly_equal(output[0], expected0));
}

void test_gemv() {
    const float input[] = {1.0F, 2.0F, 3.0F};
    const float weight[] = {
        1.0F, 0.0F, -1.0F,
        0.5F, 0.5F, 0.5F,
    };
    const float bias[] = {1.0F, -1.0F};
    float output[2] = {};

    pokitlms::gemv_f32(input, weight, bias, output, 3, 2);

    assert(nearly_equal(output[0], -1.0F));
    assert(nearly_equal(output[1], 2.0F));
    float explicit_output[2] = {};
    pokitlms::linear_f32(input, weight, bias, explicit_output, 3, 2);
    assert(nearly_equal(explicit_output[0], output[0]));
    assert(nearly_equal(explicit_output[1], output[1]));
}

void test_rope() {
    float query[] = {1.0F, 0.0F, 0.0F, 1.0F};
    float key[] = {1.0F, 0.0F};

    pokitlms::apply_rope(query, key, 2, 1, 2, 1);

    assert(nearly_equal(query[0], std::cos(1.0F)));
    assert(nearly_equal(query[1], std::sin(1.0F)));
    assert(nearly_equal(key[0], std::cos(1.0F)));
    assert(nearly_equal(key[1], std::sin(1.0F)));
}

void test_kv_cache() {
    pokitlms::KVCache cache(2, 1, 2);

    const float key[] = {1.0F, 2.0F};
    const float value[] = {3.0F, 4.0F};

    cache.append(key, value);

    assert(cache.size() == 1);
    assert(cache.capacity() == 2);
    assert(nearly_equal(cache.key_at(0)[1], 2.0F));
    assert(nearly_equal(cache.value_at(0)[0], 3.0F));

    cache.clear();
    assert(cache.size() == 0);
}

void test_attention_decode() {
    pokitlms::KVCache cache(2, 1, 2);

    const float key0[] = {1.0F, 0.0F};
    const float value0[] = {10.0F, 0.0F};
    const float key1[] = {0.0F, 1.0F};
    const float value1[] = {0.0F, 20.0F};

    cache.append(key0, value0);
    cache.append(key1, value1);

    const float query[] = {1.0F, 0.0F};
    float output[2] = {};

    pokitlms::attention_decode_f32(query, cache, output, 1, 1, 2);

    const float e0 = std::exp(1.0F / std::sqrt(2.0F));
    const float e1 = 1.0F;
    const float denom = e0 + e1;
    const float expected0 = (e0 / denom) * 10.0F;
    const float expected1 = (e1 / denom) * 20.0F;

    assert(nearly_equal(output[0], expected0, 1.0e-4F));
    assert(nearly_equal(output[1], expected1, 1.0e-4F));
}

void test_gated_delta_state_layout() {
    const float query[] = {1.0F, 0.0F};
    const float key[] = {1.0F, 0.0F};
    const float value[] = {2.0F, 4.0F};
    const float log_decay[] = {0.0F};
    const float beta[] = {0.5F};
    float state[] = {0.0F, 0.0F, 0.0F, 0.0F};
    float output[] = {0.0F, 0.0F};
    pokitlms::GatedDeltaNetScratch scratch;

    pokitlms::gated_delta_recurrent_step(query, key, value, log_decay, beta,
                                          1, 2, 2, state, output, scratch);

    const float inverse_sqrt_two = 1.0F / std::sqrt(2.0F);
    assert(nearly_equal(output[0], inverse_sqrt_two));
    assert(nearly_equal(output[1], 2.0F * inverse_sqrt_two));
    assert(nearly_equal(state[0], 1.0F));
    assert(nearly_equal(state[1], 0.0F));
    assert(nearly_equal(state[2], 2.0F));
    assert(nearly_equal(state[3], 0.0F));
}

}  // namespace

int main() {
    test_tensor_and_version();
    test_rms_norm();
    test_gemv();
    test_rope();
    test_kv_cache();
    test_attention_decode();
    test_gated_delta_state_layout();
    return 0;
}
