#include "pokitlms/model/architecture.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace pokitlms::model {
namespace {

const MetadataValue& require_value(const GgufReader& model, const std::string& key) {
    const auto found = model.metadata().find(key);
    if (found == model.metadata().end()) throw std::runtime_error("missing GGUF metadata: " + key);
    return found->second;
}

std::uint64_t as_u64(const MetadataValue& value, const std::string& key) {
    if (const auto* v = std::get_if<std::uint64_t>(&value.value)) return *v;
    if (const auto* v = std::get_if<std::int64_t>(&value.value); v && *v >= 0) {
        return static_cast<std::uint64_t>(*v);
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

bool optional_bool(const GgufReader& model, const std::string& key, bool fallback) {
    const auto found = model.metadata().find(key);
    if (found == model.metadata().end()) return fallback;
    if (const auto* value = std::get_if<bool>(&found->second.value)) return *value;
    throw std::runtime_error("GGUF metadata has wrong boolean type: " + key);
}

double as_number(const MetadataValue& value, const std::string& key) {
    if (const auto* v = std::get_if<double>(&value.value)) return *v;
    if (const auto* v = std::get_if<std::uint64_t>(&value.value)) return static_cast<double>(*v);
    if (const auto* v = std::get_if<std::int64_t>(&value.value)) return static_cast<double>(*v);
    throw std::runtime_error("GGUF metadata has wrong numeric type: " + key);
}

double require_number(const GgufReader& model, const std::string& key) {
    return as_number(require_value(model, key), key);
}

void require_positive(std::uint64_t value, const char* field) {
    if (value == 0) throw std::runtime_error(std::string("invalid zero model parameter: ") + field);
}

}  // namespace

TransformerConfig load_transformer_config(const GgufReader& model) {
    const auto architecture_it = model.metadata().find("general.architecture");
    if (architecture_it == model.metadata().end()) throw std::runtime_error("missing GGUF general.architecture");
    const auto* architecture = std::get_if<std::string>(&architecture_it->second.value);
    if (!architecture) throw std::runtime_error("GGUF general.architecture must be a string");
    if (*architecture != "qwen3moe") {
        throw std::runtime_error("unsupported model architecture: " + *architecture);
    }

    const std::string prefix = *architecture + ".";
    TransformerConfig config;
    config.architecture = *architecture;
    config.context_length = require_u64(model, prefix + "context_length");
    config.embedding_length = require_u64(model, prefix + "embedding_length");
    config.block_count = require_u64(model, prefix + "block_count");
    config.feed_forward_length = require_u64(model, prefix + "feed_forward_length");
    config.attention_heads = require_u64(model, prefix + "attention.head_count");
    require_positive(config.attention_heads, "attention_heads");
    config.key_value_heads = optional_u64(model, prefix + "attention.head_count_kv",
                                          config.attention_heads);
    config.key_length = optional_u64(model, prefix + "attention.key_length",
                                     config.embedding_length / config.attention_heads);
    config.value_length = optional_u64(model, prefix + "attention.value_length", config.key_length);
    config.rotary_dimension = optional_u64(model, prefix + "rope.dimension_count", config.key_length);
    config.expert_count = require_u64(model, prefix + "expert_count");
    config.experts_per_token = require_u64(model, prefix + "expert_used_count");
    config.expert_feed_forward_length = require_u64(model, prefix + "expert_feed_forward_length");
    config.expert_weights_norm = optional_bool(model, prefix + "expert_weights_norm", false);
    config.tie_word_embeddings = optional_bool(model, prefix + "tie_word_embeddings",
                                                model.find_tensor("output.weight") == nullptr);
    if (model.find_tensor("output.weight") == nullptr) config.tie_word_embeddings = true;
    config.rms_norm_epsilon = require_number(model, prefix + "attention.layer_norm_rms_epsilon");
    config.rope_frequency_base = require_number(model, prefix + "rope.freq_base");

    require_positive(config.context_length, "context_length");
    require_positive(config.embedding_length, "embedding_length");
    require_positive(config.block_count, "block_count");
    require_positive(config.feed_forward_length, "feed_forward_length");
    require_positive(config.attention_heads, "attention_heads");
    require_positive(config.key_value_heads, "key_value_heads");
    require_positive(config.key_length, "key_length");
    require_positive(config.value_length, "value_length");
    require_positive(config.rotary_dimension, "rotary_dimension");
    require_positive(config.expert_count, "expert_count");
    require_positive(config.experts_per_token, "experts_per_token");
    require_positive(config.expert_feed_forward_length, "expert_feed_forward_length");
    if (config.key_value_heads > config.attention_heads ||
        config.attention_heads % config.key_value_heads != 0 ||
        config.experts_per_token > config.expert_count ||
        config.rotary_dimension > config.key_length || config.rotary_dimension % 2 != 0 ||
        !std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0 ||
        !std::isfinite(config.rope_frequency_base) || config.rope_frequency_base <= 0.0) {
        throw std::runtime_error("inconsistent qwen3moe model parameters");
    }
    if (config.attention_heads > std::numeric_limits<std::uint64_t>::max() / config.key_length ||
        config.key_value_heads > std::numeric_limits<std::uint64_t>::max() / config.key_length ||
        config.attention_heads > std::numeric_limits<std::uint64_t>::max() / config.value_length ||
        config.key_value_heads > std::numeric_limits<std::uint64_t>::max() / config.value_length) {
        throw std::runtime_error("attention dimensions overflow");
    }
    return config;
}

}  // namespace pokitlms::model
