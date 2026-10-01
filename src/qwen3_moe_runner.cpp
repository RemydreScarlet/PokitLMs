#include "pokitlms/model/qwen3_moe_runner.hpp"

#include "pokitlms/model/qwen_bpe_tokenizer.hpp"
#include "pokitlms/model/tensor_linear.hpp"
#include "pokitlms/model/tensor_reader.hpp"
#include "pokitlms/ops/kv_cache.hpp"
#include "pokitlms/ops/quantized_linear.hpp"
#include "pokitlms/ops/rms_norm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace pokitlms::model {
namespace {

std::size_t as_size(std::uint64_t value, const char* label) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("model dimension exceeds address space: ") + label);
    }
    return static_cast<std::size_t>(value);
}

std::vector<float> load_vector(const std::shared_ptr<storage::ModelFile>& file,
                               const TensorInfo& tensor) {
    if (tensor.dimensions.size() != 1) {
        throw std::invalid_argument("expected a one-dimensional GGUF weight: " + tensor.name);
    }
    TensorReader reader(file, tensor);
    return reader.read_float_rows(0, 1);
}

TensorInfo expert_view(const TensorInfo& tensor, const storage::ExpertSlice& slice) {
    TensorInfo result;
    result.name = tensor.name;
    result.dimensions = {tensor.dimensions[0], tensor.dimensions[1]};
    result.type = tensor.type;
    result.file_offset = slice.offset;
    result.payload_size = slice.size;
    return result;
}

void apply_qwen_rope(std::span<float> values, std::size_t head_dimension,
                     std::size_t rotary_dimension, std::uint64_t position, float theta) {
    if (head_dimension == 0 || values.size() % head_dimension != 0 ||
        rotary_dimension == 0 || rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 ||
        !std::isfinite(theta) || theta <= 0.0F) {
        throw std::invalid_argument("invalid Qwen RoPE parameters");
    }
    const auto heads = values.size() / head_dimension;
    const auto half = rotary_dimension / 2;
    for (std::size_t head = 0; head < heads; ++head) {
        auto* vector = values.data() + head * head_dimension;
        for (std::size_t pair = 0; pair < half; ++pair) {
            const float angle = static_cast<float>(position) *
                std::pow(theta, -static_cast<float>(pair) / static_cast<float>(half));
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float first = vector[pair];
            const float second = vector[pair + half];
            vector[pair] = first * cosine - second * sine;
            vector[pair + half] = second * cosine + first * sine;
        }
    }
}

struct Route {
    std::size_t id;
    float probability;
};

std::vector<Route> route_experts(std::span<const float> logits, std::size_t top_k,
                                 bool normalize_top_k) {
    if (top_k == 0 || top_k > logits.size()) throw std::invalid_argument("invalid expert top-k");
    const auto max_it = std::max_element(logits.begin(), logits.end());
    if (max_it == logits.end() || !std::isfinite(*max_it)) {
        throw std::runtime_error("expert router produced non-finite logits");
    }
    if (std::any_of(logits.begin(), logits.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("expert router produced non-finite logits");
    }
    double denominator = 0.0;
    for (const auto value : logits) denominator += std::exp(static_cast<double>(value - *max_it));
    std::vector<std::size_t> ids(logits.size());
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i;
    std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(top_k), ids.end(),
        [&logits](std::size_t lhs, std::size_t rhs) {
            return logits[lhs] > logits[rhs] || (logits[lhs] == logits[rhs] && lhs < rhs);
        });
    std::vector<Route> routes;
    routes.reserve(top_k);
    float selected_sum = 0.0F;
    for (std::size_t i = 0; i < top_k; ++i) {
        const auto id = ids[i];
        const float probability = static_cast<float>(std::exp(
            static_cast<double>(logits[id] - *max_it)) / denominator);
        routes.push_back({id, probability});
        selected_sum += probability;
    }
    if (normalize_top_k) {
        for (auto& route : routes) route.probability /= selected_sum;
    }
    return routes;
}

std::uint32_t sample_token(std::span<const float> logits,
                           std::span<const std::uint32_t> history,
                           const GenerationOptions& options,
                           std::mt19937_64& random) {
    if (!std::isfinite(options.temperature) || options.temperature < 0.0F ||
        !std::isfinite(options.top_p) || options.top_p <= 0.0F || options.top_p > 1.0F ||
        !std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("invalid generation sampling options");
    }
    if (logits.empty() || logits.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid logits vocabulary size");
    }
    std::vector<float> adjusted(logits.begin(), logits.end());
    for (const float logit : adjusted) {
        if (!std::isfinite(logit)) throw std::runtime_error("model produced non-finite logits");
    }
    if (options.repetition_penalty != 1.0F) {
        std::vector<bool> seen(adjusted.size(), false);
        for (const auto id : history) {
            if (id >= adjusted.size()) continue;
            if (seen[id]) continue;
            seen[id] = true;
            adjusted[id] = adjusted[id] < 0.0F
                ? adjusted[id] * options.repetition_penalty
                : adjusted[id] / options.repetition_penalty;
        }
    }
    if (options.temperature == 0.0F) {
        return static_cast<std::uint32_t>(std::distance(adjusted.begin(),
            std::max_element(adjusted.begin(), adjusted.end())));
    }

    const double inverse_temperature = 1.0 / static_cast<double>(options.temperature);
    std::vector<std::pair<double, std::uint32_t>> ranked;
    ranked.reserve(adjusted.size());
    for (std::size_t id = 0; id < adjusted.size(); ++id) {
        ranked.emplace_back(static_cast<double>(adjusted[id]) * inverse_temperature,
                            static_cast<std::uint32_t>(id));
    }
    const auto order = [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    };
    if (options.top_k != 0 && options.top_k < ranked.size()) {
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(options.top_k),
                          ranked.end(), order);
        ranked.resize(options.top_k);
    }
    if (options.top_p < 1.0F) std::sort(ranked.begin(), ranked.end(), order);

    double max_logit = -std::numeric_limits<double>::infinity();
    for (const auto& item : ranked) max_logit = std::max(max_logit, item.first);
    std::vector<double> weights;
    weights.reserve(ranked.size());
    double total = 0.0;
    for (const auto& item : ranked) {
        const auto weight = std::exp(static_cast<double>(item.first - max_logit));
        weights.push_back(weight);
        total += weight;
    }
    if (options.top_p < 1.0F) {
        std::size_t keep = 0;
        double cumulative = 0.0;
        do {
            cumulative += weights[keep] / total;
            ++keep;
        } while (keep < weights.size() && cumulative < options.top_p);
        ranked.resize(keep);
        weights.resize(keep);
    }
    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return ranked[distribution(random)].second;
}

}  // namespace

class Qwen3MoeRunner::Impl {
public:
    Impl(std::filesystem::path path, std::size_t cache_budget, std::size_t requested_context)
        : model_path(std::move(path)), gguf(model_path),
          file(std::make_shared<storage::ModelFile>(model_path)), index(gguf), tokenizer(gguf) {
        const auto& config = index.config();
        const auto model_context = as_size(config.context_length, "context_length");
        // Full advertised contexts require multiple gigabytes of FP32 KV state on MoE models.
        // Start with a mobile-sized window and let callers opt into a larger allocation.
        context_capacity = requested_context == 0 ? std::min<std::size_t>(model_context, 512) : requested_context;
        if (context_capacity == 0 || context_capacity > model_context) {
            throw std::invalid_argument("requested context exceeds Qwen3-MoE model context");
        }
        const auto layer_count = as_size(config.block_count, "block_count");
        const auto key_dim = as_size(config.key_length, "key_length");
        const auto value_dim = as_size(config.value_length, "value_length");
        const auto kv_heads = as_size(config.key_value_heads, "key_value_heads");
        if (layer_count > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("model has too many layers for expert cache accounting");
        }
        const auto per_store_budget = cache_budget / std::max<std::size_t>(1, layer_count * 3);
        attention_norms.reserve(layer_count);
        query_norms.reserve(layer_count);
        key_norms.reserve(layer_count);
        feed_forward_norms.reserve(layer_count);
        kv_caches.reserve(layer_count);
        gate_stores.reserve(layer_count);
        up_stores.reserve(layer_count);
        down_stores.reserve(layer_count);
        const auto expert_count = as_size(config.expert_count, "expert_count");
        for (const auto& block : index.blocks()) {
            attention_norms.push_back(load_vector(file, block.attention_norm));
            query_norms.push_back(load_vector(file, block.query_norm));
            key_norms.push_back(load_vector(file, block.key_norm));
            feed_forward_norms.push_back(load_vector(file, block.feed_forward_norm));
            kv_caches.emplace_back(context_capacity, kv_heads, key_dim, value_dim,
                                   KvCachePrecision::Float16);
            gate_stores.push_back(std::make_unique<storage::ExpertStore>(
                file, storage::split_expert_tensor(block.expert_gate, expert_count), per_store_budget));
            up_stores.push_back(std::make_unique<storage::ExpertStore>(
                file, storage::split_expert_tensor(block.expert_up, expert_count), per_store_budget));
            down_stores.push_back(std::make_unique<storage::ExpertStore>(
                file, storage::split_expert_tensor(block.expert_down, expert_count), per_store_budget));
        }
        output_norm = load_vector(file, index.output_norm());
        next_position = 0;
    }

    std::filesystem::path model_path;
    GgufReader gguf;
    std::shared_ptr<storage::ModelFile> file;
    Qwen3MoeIndex index;
    QwenBpeTokenizer tokenizer;
    std::size_t context_capacity{};
    std::uint64_t next_position{};
    std::vector<float> output_norm;
    std::vector<std::vector<float>> attention_norms;
    std::vector<std::vector<float>> query_norms;
    std::vector<std::vector<float>> key_norms;
    std::vector<std::vector<float>> feed_forward_norms;
    std::vector<KvCache> kv_caches;
    std::vector<std::unique_ptr<storage::ExpertStore>> gate_stores;
    std::vector<std::unique_ptr<storage::ExpertStore>> up_stores;
    std::vector<std::unique_ptr<storage::ExpertStore>> down_stores;

    std::vector<float> forward(std::uint32_t token_id, std::uint64_t position) {
        const auto& config = index.config();
        if (token_id >= index.vocabulary_size()) throw std::out_of_range("token id exceeds model vocabulary");
        if (position != next_position || position >= context_capacity) {
            throw std::invalid_argument("token position is not the next position in this decode state");
        }
        try {
            TensorReader embeddings(file, index.token_embedding());
            std::vector<float> hidden;
            if (index.token_embedding().type == 0 || index.token_embedding().type == 1) {
                hidden = embeddings.read_float_rows(token_id, 1);
            } else {
                const auto encoded = embeddings.read_rows(token_id, 1);
                hidden.resize(as_size(config.embedding_length, "embedding_length"));
                dequantize_quantized_row(index.token_embedding().type, encoded, hidden);
            }
            const auto hidden_size = hidden.size();
            const auto query_heads = as_size(config.attention_heads, "attention_heads");
            const auto kv_heads = as_size(config.key_value_heads, "key_value_heads");
            const auto key_dim = as_size(config.key_length, "key_length");
            const auto value_dim = as_size(config.value_length, "value_length");
            const auto rotary_dim = as_size(config.rotary_dimension, "rotary_dimension");
            const auto q_size = query_heads * key_dim;
            const auto k_size = kv_heads * key_dim;
            const auto v_size = kv_heads * value_dim;
            const auto epsilon = static_cast<float>(config.rms_norm_epsilon);
            const auto theta = static_cast<float>(config.rope_frequency_base);
            for (std::size_t layer = 0; layer < index.blocks().size(); ++layer) {
                const auto& block = index.blocks()[layer];
                std::vector<float> normalized(hidden_size);
                rms_norm(hidden, attention_norms[layer], normalized, epsilon);
                std::vector<float> query(q_size), key(k_size), value(v_size);
                tensor_linear(TensorReader(file, block.query), normalized, query);
                tensor_linear(TensorReader(file, block.key), normalized, key);
                tensor_linear(TensorReader(file, block.value), normalized, value);
                for (std::size_t head = 0; head < query_heads; ++head) {
                    auto* begin = query.data() + head * key_dim;
                    rms_norm(std::span<const float>(begin, key_dim), query_norms[layer],
                             std::span<float>(begin, key_dim), epsilon);
                }
                for (std::size_t head = 0; head < kv_heads; ++head) {
                    auto* begin = key.data() + head * key_dim;
                    rms_norm(std::span<const float>(begin, key_dim), key_norms[layer],
                             std::span<float>(begin, key_dim), epsilon);
                }
                apply_qwen_rope(query, key_dim, rotary_dim, position, theta);
                apply_qwen_rope(key, key_dim, rotary_dim, position, theta);
                kv_caches[layer].append(position, key.data(), key.size(), value.data(), value.size());
                std::vector<float> attended(query_heads * value_dim);
                kv_caches[layer].attend(query.data(), query.size(), attended.data(), attended.size());
                std::vector<float> attention_output(hidden_size);
                tensor_linear(TensorReader(file, block.attention_output), attended, attention_output);
                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += attention_output[i];

                rms_norm(hidden, feed_forward_norms[layer], normalized, epsilon);
                const auto expert_count = as_size(config.expert_count, "expert_count");
                std::vector<float> router_logits(expert_count);
                tensor_linear(TensorReader(file, block.router), normalized, router_logits);
                const auto routes = route_experts(router_logits,
                    as_size(config.experts_per_token, "experts_per_token"), config.expert_weights_norm);
                std::vector<float> moe_result(hidden_size, 0.0F);
                const auto intermediate = as_size(config.expert_feed_forward_length,
                                                  "expert_feed_forward_length");
                const auto gate_slices = storage::split_expert_tensor(block.expert_gate, expert_count);
                const auto up_slices = storage::split_expert_tensor(block.expert_up, expert_count);
                const auto down_slices = storage::split_expert_tensor(block.expert_down, expert_count);
                for (const auto& route : routes) {
                    const auto gate_bytes = gate_stores[layer]->get(route.id);
                    const auto up_bytes = up_stores[layer]->get(route.id);
                    const auto down_bytes = down_stores[layer]->get(route.id);
                    TensorReader gate_reader(expert_view(block.expert_gate, gate_slices[route.id]), gate_bytes);
                    TensorReader up_reader(expert_view(block.expert_up, up_slices[route.id]), up_bytes);
                    TensorReader down_reader(expert_view(block.expert_down, down_slices[route.id]), down_bytes);
                    std::vector<float> gate(intermediate), up(intermediate), activated(intermediate);
                    tensor_linear(gate_reader, normalized, gate);
                    tensor_linear(up_reader, normalized, up);
                    for (std::size_t i = 0; i < intermediate; ++i) {
                        activated[i] = (gate[i] / (1.0F + std::exp(-gate[i]))) * up[i];
                    }
                    std::vector<float> expert_output(hidden_size);
                    tensor_linear(down_reader, activated, expert_output);
                    for (std::size_t i = 0; i < hidden_size; ++i) {
                        moe_result[i] += route.probability * expert_output[i];
                    }
                }
                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += moe_result[i];
            }
            std::vector<float> final_hidden(hidden_size);
            rms_norm(hidden, output_norm, final_hidden, epsilon);
            std::vector<float> logits(as_size(index.vocabulary_size(), "vocabulary_size"));
            tensor_linear(TensorReader(file, index.output()), final_hidden, logits);
            ++next_position;
            return logits;
        } catch (...) {
            for (auto& cache : kv_caches) cache.clear();
            next_position = 0;
            throw;
        }
    }

    void reset() {
        for (auto& cache : kv_caches) cache.clear();
        next_position = 0;
    }

    std::vector<std::uint32_t> generate(std::span<const std::uint32_t> prompt,
                                        std::size_t max_new_tokens,
                                        const GenerationOptions& options) {
        reset();
        if (prompt.empty()) throw std::invalid_argument("generation prompt must contain at least one token");
        if (prompt.size() > context_capacity || max_new_tokens > context_capacity - prompt.size()) {
            throw std::length_error("prompt and generation length exceed KV context capacity");
        }
        std::vector<std::uint32_t> history(prompt.begin(), prompt.end());
        std::vector<std::uint32_t> generated;
        if (max_new_tokens == 0) return generated;
        std::vector<float> logits;
        try {
            for (std::size_t i = 0; i < prompt.size(); ++i) {
                logits = forward(prompt[i], i);
            }
            std::mt19937_64 random(options.seed);
            generated.reserve(max_new_tokens);
            for (std::size_t i = 0; i < max_new_tokens; ++i) {
                const auto token = sample_token(logits, history, options, random);
                generated.push_back(token);
                history.push_back(token);
                if (token == tokenizer.eos_token_id()) break;
                if (i + 1 < max_new_tokens) logits = forward(token, prompt.size() + i);
            }
        } catch (...) {
            reset();
            throw;
        }
        return generated;
    }
};

Qwen3MoeRunner::Qwen3MoeRunner(std::filesystem::path model_path,
                               std::size_t expert_cache_budget_bytes,
                               std::size_t context_capacity) {
    impl_ = std::make_unique<Impl>(std::move(model_path), expert_cache_budget_bytes,
                                   context_capacity);
}

Qwen3MoeRunner::~Qwen3MoeRunner() = default;
Qwen3MoeRunner::Qwen3MoeRunner(Qwen3MoeRunner&&) noexcept = default;
Qwen3MoeRunner& Qwen3MoeRunner::operator=(Qwen3MoeRunner&&) noexcept = default;

std::vector<float> Qwen3MoeRunner::forward_token(std::uint32_t token_id, std::uint64_t position) {
    if (!impl_) throw std::logic_error("Qwen3-MoE runner has been moved from");
    return impl_->forward(token_id, position);
}
std::vector<std::uint32_t> Qwen3MoeRunner::generate_tokens(
    std::span<const std::uint32_t> prompt, std::size_t max_new_tokens,
    const GenerationOptions& options) {
    if (!impl_) throw std::logic_error("Qwen3-MoE runner has been moved from");
    return impl_->generate(prompt, max_new_tokens, options);
}
std::string Qwen3MoeRunner::generate_text(std::string_view prompt, std::size_t max_new_tokens,
                                          const GenerationOptions& options) {
    if (!impl_) throw std::logic_error("Qwen3-MoE runner has been moved from");
    auto prompt_tokens = impl_->tokenizer.encode(std::string(prompt));
    const auto generated = impl_->generate(prompt_tokens, max_new_tokens, options);
    std::vector<std::uint32_t> printable;
    printable.reserve(generated.size());
    for (const auto token : generated) {
        if (token == impl_->tokenizer.eos_token_id()) break;
        printable.push_back(token);
    }
    return impl_->tokenizer.decode(printable);
}
void Qwen3MoeRunner::reset() { if (impl_) impl_->reset(); }
const TransformerConfig& Qwen3MoeRunner::config() const noexcept {
    static const TransformerConfig empty{};
    return impl_ ? impl_->index.config() : empty;
}
std::uint64_t Qwen3MoeRunner::bytes_read_from_disk() const noexcept {
    return impl_ && impl_->file ? impl_->file->bytes_read() : 0;
}

}  // namespace pokitlms::model
