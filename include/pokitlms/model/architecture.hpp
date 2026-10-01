#pragma once

#include "pokitlms/model/gguf_reader.hpp"

#include <cstdint>
#include <string>

namespace pokitlms::model {

struct TransformerConfig {
    std::string architecture;
    std::uint64_t context_length{};
    std::uint64_t embedding_length{};
    std::uint64_t block_count{};
    std::uint64_t feed_forward_length{};
    std::uint64_t attention_heads{};
    std::uint64_t key_value_heads{};
    std::uint64_t key_length{};
    std::uint64_t value_length{};
    std::uint64_t rotary_dimension{};
    std::uint64_t expert_count{};
    std::uint64_t experts_per_token{};
    std::uint64_t expert_feed_forward_length{};
    bool expert_weights_norm{};
    double rms_norm_epsilon{};
    double rope_frequency_base{};
};

// Loads validated architecture parameters for model families implemented by
// this runtime. Unknown families and incomplete metadata are rejected early.
[[nodiscard]] TransformerConfig load_transformer_config(const GgufReader& model);

}  // namespace pokitlms::model
