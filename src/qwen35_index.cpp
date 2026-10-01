#include "pokitlms/model/qwen35_index.hpp"

#include <initializer_list>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace pokitlms::model {
namespace {

const MetadataValue& require_value(const GgufReader& model, const std::string& key) {
    const auto found = model.metadata().find(key);
    if (found == model.metadata().end()) throw std::runtime_error("missing GGUF metadata: " + key);
    return found->second;
}

std::uint64_t as_u64(const MetadataValue& value, const std::string& key) {
    if (const auto* number = std::get_if<std::uint64_t>(&value.value)) return *number;
    if (const auto* number = std::get_if<std::int64_t>(&value.value); number && *number >= 0) {
        return static_cast<std::uint64_t>(*number);
    }
    throw std::runtime_error("GGUF metadata has wrong integer type: " + key);
}

std::uint64_t require_u64(const GgufReader& model, const std::string& key) {
    return as_u64(require_value(model, key), key);
}

std::uint64_t optional_u64(const GgufReader& model, const std::string& key,
                           std::uint64_t fallback) {
    const auto found = model.metadata().find(key);
    return found == model.metadata().end() ? fallback : as_u64(found->second, key);
}

double require_number(const GgufReader& model, const std::string& key) {
    const auto& value = require_value(model, key);
    if (const auto* number = std::get_if<double>(&value.value)) return *number;
    if (const auto* number = std::get_if<std::uint64_t>(&value.value)) return static_cast<double>(*number);
    if (const auto* number = std::get_if<std::int64_t>(&value.value)) return static_cast<double>(*number);
    throw std::runtime_error("GGUF metadata has wrong numeric type: " + key);
}

bool optional_bool(const GgufReader& model, const std::string& key, bool fallback) {
    const auto found = model.metadata().find(key);
    if (found == model.metadata().end()) return fallback;
    if (const auto* value = std::get_if<bool>(&found->second.value)) return *value;
    throw std::runtime_error("GGUF metadata has wrong boolean type: " + key);
}

void require_positive(std::uint64_t value, const char* name) {
    if (value == 0) throw std::runtime_error(std::string("invalid zero Qwen3.5 parameter: ") + name);
}

std::uint64_t checked_product(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
        throw std::runtime_error(std::string("Qwen3.5 dimension overflow: ") + name);
    }
    return lhs * rhs;
}

const TensorInfo& require_tensor(const GgufReader& model, const std::string& name,
                                std::initializer_list<std::uint64_t> dimensions) {
    const auto* tensor = model.find_tensor(name);
    if (!tensor) throw std::runtime_error("missing Qwen3.5 tensor: " + name);
    if (tensor->dimensions != std::vector<std::uint64_t>(dimensions) || !tensor->payload_size) {
        throw std::runtime_error("unexpected dimensions or storage type for tensor: " + name);
    }
    return *tensor;
}

std::uint64_t vocab_size(const GgufReader& model) {
    const auto found = model.metadata().find("tokenizer.ggml.tokens");
    if (found == model.metadata().end()) throw std::runtime_error("missing tokenizer.ggml.tokens");
    const auto* tokens = std::get_if<MetadataValue::Array>(&found->second.value);
    if (!tokens || tokens->empty()) throw std::runtime_error("invalid tokenizer.ggml.tokens");
    for (const auto& token : *tokens) {
        if (!std::holds_alternative<std::string>(token.value)) {
            throw std::runtime_error("tokenizer.ggml.tokens contains a non-string value");
        }
    }
    return tokens->size();
}

Qwen35Config load_config(const GgufReader& model) {
    const auto arch = model.metadata().find("general.architecture");
    if (arch == model.metadata().end()) throw std::runtime_error("missing GGUF general.architecture");
    const auto* name = std::get_if<std::string>(&arch->second.value);
    if (!name || *name != "qwen35") throw std::runtime_error("GGUF is not a Qwen3.5 text model");

    constexpr std::string_view prefix = "qwen35.";
    const auto key = [prefix](std::string_view suffix) {
        return std::string(prefix) + std::string(suffix);
    };
    Qwen35Config config;
    config.context_length = require_u64(model, key("context_length"));
    config.embedding_length = require_u64(model, key("embedding_length"));
    config.total_block_count = require_u64(model, key("block_count"));
    config.feed_forward_length = require_u64(model, key("feed_forward_length"));
    config.attention_heads = require_u64(model, key("attention.head_count"));
    config.key_value_heads = require_u64(model, key("attention.head_count_kv"));
    config.attention_head_length = require_u64(model, key("attention.key_length"));
    config.value_head_length = optional_u64(model, key("attention.value_length"),
                                             config.attention_head_length);
    config.rotary_dimension = require_u64(model, key("rope.dimension_count"));
    config.ssm_conv_kernel = require_u64(model, key("ssm.conv_kernel"));
    config.ssm_state_size = require_u64(model, key("ssm.state_size"));
    config.ssm_group_count = require_u64(model, key("ssm.group_count"));
    config.ssm_time_step_rank = require_u64(model, key("ssm.time_step_rank"));
    config.ssm_inner_size = require_u64(model, key("ssm.inner_size"));
    config.full_attention_interval = require_u64(model, key("full_attention_interval"));
    config.nextn_predict_layers = optional_u64(model, key("nextn_predict_layers"), 0);
    config.rms_norm_epsilon = require_number(model, key("attention.layer_norm_rms_epsilon"));
    config.rope_frequency_base = require_number(model, key("rope.freq_base"));
    config.tie_word_embeddings = optional_bool(model, key("tie_word_embeddings"),
                                                model.find_tensor("output.weight") == nullptr);
    if (model.find_tensor("output.weight") == nullptr) config.tie_word_embeddings = true;

    require_positive(config.context_length, "context_length");
    require_positive(config.embedding_length, "embedding_length");
    require_positive(config.total_block_count, "block_count");
    require_positive(config.feed_forward_length, "feed_forward_length");
    require_positive(config.attention_heads, "attention.head_count");
    require_positive(config.key_value_heads, "attention.head_count_kv");
    require_positive(config.attention_head_length, "attention.key_length");
    require_positive(config.value_head_length, "attention.value_length");
    require_positive(config.rotary_dimension, "rope.dimension_count");
    require_positive(config.ssm_conv_kernel, "ssm.conv_kernel");
    require_positive(config.ssm_state_size, "ssm.state_size");
    require_positive(config.ssm_group_count, "ssm.group_count");
    require_positive(config.ssm_time_step_rank, "ssm.time_step_rank");
    require_positive(config.ssm_inner_size, "ssm.inner_size");
    require_positive(config.full_attention_interval, "full_attention_interval");
    if (config.nextn_predict_layers >= config.total_block_count ||
        config.attention_heads % config.key_value_heads != 0 ||
        config.attention_head_length != config.value_head_length ||
        config.rotary_dimension > config.attention_head_length ||
        config.rotary_dimension % 2 != 0 ||
        config.ssm_inner_size % config.ssm_time_step_rank != 0 ||
        config.full_attention_interval > config.total_block_count ||
        !std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0 ||
        !std::isfinite(config.rope_frequency_base) || config.rope_frequency_base <= 0.0) {
        throw std::runtime_error("inconsistent Qwen3.5 model parameters");
    }
    config.block_count = config.total_block_count - config.nextn_predict_layers;
    if (checked_product(config.ssm_group_count, config.ssm_state_size, "linear key dimension") >
        std::numeric_limits<std::uint64_t>::max() / 2) {
        throw std::runtime_error("Qwen3.5 linear QKV dimension overflows");
    }
    return config;
}

}  // namespace

bool Qwen35Config::is_full_attention_layer(std::uint64_t layer) const noexcept {
    return full_attention_interval != 0 && (layer + 1) % full_attention_interval == 0;
}

std::uint64_t Qwen35Config::linear_key_dimension() const noexcept {
    return ssm_group_count * ssm_state_size;
}

std::uint64_t Qwen35Config::linear_value_head_dimension() const noexcept {
    return ssm_inner_size / ssm_time_step_rank;
}

std::uint64_t Qwen35Config::linear_value_head_count() const noexcept {
    return ssm_time_step_rank;
}

Qwen35Index::Qwen35Index(const GgufReader& model)
    : config_(load_config(model)), vocabulary_size_(vocab_size(model)) {
    token_embedding_ = require_tensor(model, "token_embd.weight",
                                      {config_.embedding_length, vocabulary_size_});
    output_norm_ = require_tensor(model, "output_norm.weight", {config_.embedding_length});
    if (const auto* output = model.find_tensor("output.weight")) {
        if (output->dimensions != std::vector<std::uint64_t>{config_.embedding_length,
                                                              vocabulary_size_} ||
            !output->payload_size) {
            throw std::runtime_error("unexpected dimensions or storage type for tensor: output.weight");
        }
        output_ = *output;
    } else if (config_.tie_word_embeddings) {
        output_ = token_embedding_;
    } else {
        throw std::runtime_error("missing Qwen3.5 output.weight");
    }

    const auto hidden = config_.embedding_length;
    const auto ffn = config_.feed_forward_length;
    const auto key_heads = config_.attention_heads;
    const auto kv_heads = config_.key_value_heads;
    const auto head_dim = config_.attention_head_length;
    const auto q_dimension = checked_product(key_heads, head_dim, "attention query");
    const auto kv_dimension = checked_product(kv_heads, head_dim, "attention key/value");
    const auto query_tensor_dimension = checked_product(q_dimension, 2, "gated attention query");
    const auto linear_key_dim = config_.linear_key_dimension();
    const auto linear_value_dim = config_.ssm_inner_size;
    const auto doubled_linear_key_dim = checked_product(2, linear_key_dim, "linear QKV");
    if (linear_value_dim > std::numeric_limits<std::uint64_t>::max() - doubled_linear_key_dim) {
        throw std::runtime_error("Qwen3.5 linear QKV dimension overflows");
    }
    const auto linear_conv_dim = doubled_linear_key_dim + linear_value_dim;
    if (linear_conv_dim > std::numeric_limits<std::uint64_t>::max() / config_.ssm_conv_kernel) {
        throw std::runtime_error("Qwen3.5 convolution tensor dimensions overflow");
    }
    if (config_.block_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Qwen3.5 layer count exceeds address space");
    }

    layers_.reserve(static_cast<std::size_t>(config_.block_count));
    for (std::uint64_t layer = 0; layer < config_.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        Qwen35LayerTensors tensors;
        tensors.full_attention = config_.is_full_attention_layer(layer);
        tensors.attention_norm = require_tensor(model, prefix + "attn_norm.weight", {hidden});
        tensors.post_attention_norm = require_tensor(model, prefix + "post_attention_norm.weight", {hidden});
        tensors.feed_forward_gate = require_tensor(model, prefix + "ffn_gate.weight", {hidden, ffn});
        tensors.feed_forward_up = require_tensor(model, prefix + "ffn_up.weight", {hidden, ffn});
        tensors.feed_forward_down = require_tensor(model, prefix + "ffn_down.weight", {ffn, hidden});
        if (tensors.full_attention) {
            tensors.query = require_tensor(model, prefix + "attn_q.weight",
                                            {hidden, query_tensor_dimension});
            tensors.key = require_tensor(model, prefix + "attn_k.weight", {hidden, kv_dimension});
            tensors.value = require_tensor(model, prefix + "attn_v.weight", {hidden, kv_dimension});
            tensors.query_norm = require_tensor(model, prefix + "attn_q_norm.weight", {head_dim});
            tensors.key_norm = require_tensor(model, prefix + "attn_k_norm.weight", {head_dim});
            tensors.attention_output = require_tensor(model, prefix + "attn_output.weight",
                {checked_product(key_heads, config_.value_head_length, "attention output"), hidden});
        } else {
            tensors.linear_qkv = require_tensor(model, prefix + "attn_qkv.weight",
                                                 {hidden, linear_conv_dim});
            tensors.linear_gate = require_tensor(model, prefix + "attn_gate.weight",
                                                  {hidden, linear_value_dim});
            tensors.linear_a = require_tensor(model, prefix + "ssm_a", {config_.ssm_time_step_rank});
            tensors.linear_alpha = require_tensor(model, prefix + "ssm_alpha.weight",
                                                   {hidden, config_.ssm_time_step_rank});
            tensors.linear_beta = require_tensor(model, prefix + "ssm_beta.weight",
                                                  {hidden, config_.ssm_time_step_rank});
            tensors.linear_convolution = require_tensor(model, prefix + "ssm_conv1d.weight",
                {config_.ssm_conv_kernel, linear_conv_dim});
            tensors.linear_dt_bias = require_tensor(model, prefix + "ssm_dt.bias",
                                                     {config_.ssm_time_step_rank});
            tensors.linear_norm = require_tensor(model, prefix + "ssm_norm.weight",
                                                  {config_.linear_value_head_dimension()});
            tensors.linear_output = require_tensor(model, prefix + "ssm_out.weight",
                                                    {linear_value_dim, hidden});
        }
        layers_.push_back(std::move(tensors));
    }
}

const Qwen35Config& Qwen35Index::config() const noexcept { return config_; }
std::uint64_t Qwen35Index::vocabulary_size() const noexcept { return vocabulary_size_; }
const TensorInfo& Qwen35Index::token_embedding() const noexcept { return token_embedding_; }
const TensorInfo& Qwen35Index::output_norm() const noexcept { return output_norm_; }
const TensorInfo& Qwen35Index::output() const noexcept { return output_; }
const std::vector<Qwen35LayerTensors>& Qwen35Index::layers() const noexcept { return layers_; }

}  // namespace pokitlms::model
