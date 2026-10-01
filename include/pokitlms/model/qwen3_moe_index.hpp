#pragma once

#include "pokitlms/model/architecture.hpp"

#include <vector>

namespace pokitlms::model {

struct Qwen3MoeBlockTensors {
    TensorInfo attention_norm;
    TensorInfo query;
    TensorInfo query_norm;
    TensorInfo key;
    TensorInfo key_norm;
    TensorInfo value;
    TensorInfo attention_output;
    TensorInfo feed_forward_norm;
    TensorInfo router;
    TensorInfo expert_gate;
    TensorInfo expert_up;
    TensorInfo expert_down;
};

// Validates the Qwen3-MoE tensor naming/layout contract and retains only its
// tensor directory entries. Weight payloads remain untouched on disk.
class Qwen3MoeIndex {
public:
    explicit Qwen3MoeIndex(const GgufReader& model);

    [[nodiscard]] const TransformerConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t vocabulary_size() const noexcept;
    [[nodiscard]] const TensorInfo& token_embedding() const noexcept;
    [[nodiscard]] const TensorInfo& output_norm() const noexcept;
    [[nodiscard]] const TensorInfo& output() const noexcept;
    [[nodiscard]] const std::vector<Qwen3MoeBlockTensors>& blocks() const noexcept;

private:
    TransformerConfig config_;
    std::uint64_t vocabulary_size_{};
    TensorInfo token_embedding_;
    TensorInfo output_norm_;
    TensorInfo output_;
    std::vector<Qwen3MoeBlockTensors> blocks_;
};

}  // namespace pokitlms::model
