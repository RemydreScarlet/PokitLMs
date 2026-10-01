#include "pokitlms/model/qwen3_moe_index.hpp"

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace pokitlms::model {
namespace {

const TensorInfo& require_tensor(const GgufReader& model, const std::string& name,
                                 std::initializer_list<std::uint64_t> dimensions) {
    const auto* tensor = model.find_tensor(name);
    if (!tensor) throw std::runtime_error("missing Qwen3-MoE tensor: " + name);
    if (tensor->dimensions != std::vector<std::uint64_t>(dimensions)) {
        throw std::runtime_error("unexpected dimensions for tensor: " + name);
    }
    if (!tensor->payload_size) throw std::runtime_error("unsupported tensor storage type: " + name);
    return *tensor;
}

std::uint64_t get_vocab_size(const GgufReader& model) {
    const auto it = model.metadata().find("tokenizer.ggml.tokens");
    if (it == model.metadata().end()) throw std::runtime_error("missing tokenizer.ggml.tokens");
    const auto* tokens = std::get_if<MetadataValue::Array>(&it->second.value);
    if (!tokens || tokens->empty()) throw std::runtime_error("invalid tokenizer.ggml.tokens");
    for (const auto& token : *tokens) {
        if (!std::holds_alternative<std::string>(token.value)) {
            throw std::runtime_error("tokenizer.ggml.tokens contains a non-string value");
        }
    }
    return tokens->size();
}

}  // namespace

Qwen3MoeIndex::Qwen3MoeIndex(const GgufReader& model)
    : config_(load_transformer_config(model)), vocabulary_size_(get_vocab_size(model)) {
    token_embedding_ = require_tensor(model, "token_embd.weight",
                                      {config_.embedding_length, vocabulary_size_});
    output_norm_ = require_tensor(model, "output_norm.weight", {config_.embedding_length});
    output_ = require_tensor(model, "output.weight",
                             {config_.embedding_length, vocabulary_size_});

    const auto query_dimension = config_.attention_heads * config_.key_length;
    const auto key_dimension = config_.key_value_heads * config_.key_length;
    const auto value_dimension = config_.key_value_heads * config_.value_length;
    if (config_.block_count > model.tensors().size() / 12 ||
        config_.block_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Qwen3-MoE block count exceeds the available tensor directory");
    }
    blocks_.reserve(static_cast<std::size_t>(config_.block_count));
    for (std::uint64_t layer = 0; layer < config_.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto expert_shape = {config_.embedding_length, config_.expert_feed_forward_length,
                                   config_.expert_count};
        const auto down_shape = {config_.expert_feed_forward_length, config_.embedding_length,
                                 config_.expert_count};
        blocks_.push_back({
            require_tensor(model, prefix + "attn_norm.weight", {config_.embedding_length}),
            require_tensor(model, prefix + "attn_q.weight", {config_.embedding_length, query_dimension}),
            require_tensor(model, prefix + "attn_q_norm.weight", {config_.key_length}),
            require_tensor(model, prefix + "attn_k.weight", {config_.embedding_length, key_dimension}),
            require_tensor(model, prefix + "attn_k_norm.weight", {config_.key_length}),
            require_tensor(model, prefix + "attn_v.weight", {config_.embedding_length, value_dimension}),
            require_tensor(model, prefix + "attn_output.weight", {query_dimension, config_.embedding_length}),
            require_tensor(model, prefix + "ffn_norm.weight", {config_.embedding_length}),
            require_tensor(model, prefix + "ffn_gate_inp.weight", {config_.embedding_length, config_.expert_count}),
            require_tensor(model, prefix + "ffn_gate_exps.weight", expert_shape),
            require_tensor(model, prefix + "ffn_up_exps.weight", expert_shape),
            require_tensor(model, prefix + "ffn_down_exps.weight", down_shape),
        });
    }
}

const TransformerConfig& Qwen3MoeIndex::config() const noexcept { return config_; }
std::uint64_t Qwen3MoeIndex::vocabulary_size() const noexcept { return vocabulary_size_; }
const TensorInfo& Qwen3MoeIndex::token_embedding() const noexcept { return token_embedding_; }
const TensorInfo& Qwen3MoeIndex::output_norm() const noexcept { return output_norm_; }
const TensorInfo& Qwen3MoeIndex::output() const noexcept { return output_; }
const std::vector<Qwen3MoeBlockTensors>& Qwen3MoeIndex::blocks() const noexcept { return blocks_; }

}  // namespace pokitlms::model
