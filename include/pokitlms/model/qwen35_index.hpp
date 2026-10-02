#pragma once

#include "pokitlms/model/gguf_reader.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace pokitlms::model {

struct Qwen35Config {
    std::uint64_t context_length{};
    std::uint64_t embedding_length{};
    std::uint64_t total_block_count{};
    std::uint64_t block_count{};  // Excludes trailing MTP prediction layers.
    std::uint64_t feed_forward_length{};
    std::uint64_t attention_heads{};
    std::uint64_t key_value_heads{};
    std::uint64_t attention_head_length{};
    std::uint64_t value_head_length{};
    std::uint64_t rotary_dimension{};
    std::array<std::uint64_t, 4> rope_dimension_sections{};
    std::uint64_t ssm_conv_kernel{};
    std::uint64_t ssm_state_size{};
    std::uint64_t ssm_group_count{};
    std::uint64_t ssm_time_step_rank{};
    std::uint64_t ssm_inner_size{};
    std::uint64_t full_attention_interval{};
    std::uint64_t nextn_predict_layers{};
    double rms_norm_epsilon{};
    double rope_frequency_base{};
    bool tie_word_embeddings{};

    [[nodiscard]] bool is_full_attention_layer(std::uint64_t layer) const noexcept;
    [[nodiscard]] std::uint64_t linear_key_dimension() const noexcept;
    [[nodiscard]] std::uint64_t linear_key_head_count() const noexcept;
    [[nodiscard]] std::uint64_t linear_key_head_dimension() const noexcept;
    [[nodiscard]] std::uint64_t linear_value_head_dimension() const noexcept;
    [[nodiscard]] std::uint64_t linear_value_head_count() const noexcept;
};

struct Qwen35LayerTensors {
    bool full_attention{};
    TensorInfo attention_norm;
    TensorInfo post_attention_norm;
    TensorInfo feed_forward_gate;
    TensorInfo feed_forward_up;
    TensorInfo feed_forward_down;

    TensorInfo query;
    TensorInfo key;
    TensorInfo value;
    TensorInfo query_norm;
    TensorInfo key_norm;
    TensorInfo attention_output;

    TensorInfo linear_qkv;
    TensorInfo linear_gate;
    TensorInfo linear_a;
    TensorInfo linear_alpha;
    TensorInfo linear_beta;
    TensorInfo linear_convolution;
    TensorInfo linear_dt_bias;
    TensorInfo linear_norm;
    TensorInfo linear_output;
};

// Indexes the text decoder tensors of Qwen3.5 GGUFs. Trailing nextn predictor
// layers are excluded from the ordinary token-by-token decoder layer list.
class Qwen35Index {
public:
    explicit Qwen35Index(const GgufReader& model);

    [[nodiscard]] const Qwen35Config& config() const noexcept;
    [[nodiscard]] std::uint64_t vocabulary_size() const noexcept;
    [[nodiscard]] const TensorInfo& token_embedding() const noexcept;
    [[nodiscard]] const TensorInfo& output_norm() const noexcept;
    [[nodiscard]] const TensorInfo& output() const noexcept;
    [[nodiscard]] const std::vector<Qwen35LayerTensors>& layers() const noexcept;

private:
    Qwen35Config config_;
    std::uint64_t vocabulary_size_{};
    TensorInfo token_embedding_;
    TensorInfo output_norm_;
    TensorInfo output_;
    std::vector<Qwen35LayerTensors> layers_;
};

}  // namespace pokitlms::model
