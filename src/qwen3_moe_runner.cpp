#include "pokitlms/model/qwen3_moe_runner.hpp"

#include "pokitlms/model/qwen_bpe_tokenizer.hpp"
#include "pokitlms/model/tensor_linear.hpp"
#include "pokitlms/model/tensor_reader.hpp"
#include "pokitlms/ops/kv_cache.hpp"
#include "pokitlms/ops/quantized_linear.hpp"
#include "pokitlms/ops/rms_norm.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace pokitlms::model {
namespace {

std::size_t as_size(std::uint64_t value, const char* label) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("model dimension exceeds address space: ") + label);
    }
    return static_cast<std::size_t>(value);
}

std::vector<std::size_t> allocate_expert_cache_budgets(
    const std::vector<std::size_t>& slice_sizes, std::size_t total_budget) {
    std::vector<std::size_t> capacities(slice_sizes.size(), 0);
    std::vector<std::size_t> order(slice_sizes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&slice_sizes](std::size_t lhs, std::size_t rhs) {
        return slice_sizes[lhs] < slice_sizes[rhs] ||
               (slice_sizes[lhs] == slice_sizes[rhs] && lhs < rhs);
    });

    std::size_t remaining = total_budget;
    std::vector<std::size_t> active;
    for (const auto store : order) {
        const auto bytes = slice_sizes[store];
        if (bytes == 0 || bytes > remaining) continue;
        capacities[store] = bytes;
        remaining -= bytes;
        active.push_back(store);
    }

    // Give each cache one entry first, then add whole-entry slots in rounds.
    // This preserves a hard aggregate budget without disabling large experts
    // just because an equal per-store fraction is too small to admit one.
    bool allocated = true;
    while (allocated) {
        allocated = false;
        for (const auto store : active) {
            const auto bytes = slice_sizes[store];
            if (bytes <= remaining) {
                capacities[store] += bytes;
                remaining -= bytes;
                allocated = true;
            }
        }
    }
    return capacities;
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
    float logit;
    float probability;
};

struct ExpertWeights {
    std::shared_ptr<const std::vector<std::byte>> gate;
    std::shared_ptr<const std::vector<std::byte>> up;
    std::shared_ptr<const std::vector<std::byte>> down;
};

class ExpertLoader {
public:
    ExpertLoader() {
        workers_.reserve(kWorkerCount);
        try {
            for (std::size_t i = 0; i < kWorkerCount; ++i) {
                workers_.emplace_back([this] { run(); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (auto& worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }
    ~ExpertLoader() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    ExpertLoader(const ExpertLoader&) = delete;
    ExpertLoader& operator=(const ExpertLoader&) = delete;

    template <typename Work>
    [[nodiscard]] auto submit(Work&& work) {
        using Result = std::invoke_result_t<Work>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Work>(work));
        auto result = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) throw std::logic_error("expert loader is stopping");
            tasks_.emplace_back([task] { (*task)(); });
        }
        ready_.notify_one();
        return result;
    }

private:
    void run() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty() && stopping_) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    // One worker executes the expert batch and reads gate weights directly;
    // the other two workers read up/down weights concurrently.
    static constexpr std::size_t kWorkerCount = 3;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> tasks_;
    bool stopping_{};
    std::vector<std::thread> workers_;
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
    std::vector<Route> selected;
    selected.reserve(top_k);
    for (std::size_t id = 0; id < logits.size(); ++id) {
        const Route candidate{id, logits[id], 0.0F};
        const auto where = std::lower_bound(selected.begin(), selected.end(), candidate,
            [](const Route& lhs, const Route& rhs) {
                return lhs.logit > rhs.logit ||
                       (lhs.logit == rhs.logit && lhs.id < rhs.id);
            });
        if (selected.size() < top_k) selected.insert(where, candidate);
        else if (where != selected.end()) {
            selected.insert(where, candidate);
            selected.pop_back();
        }
    }
    float selected_sum = 0.0F;
    for (std::size_t i = 0; i < top_k; ++i) {
        const auto id = selected[i].id;
        selected[i].probability = static_cast<float>(std::exp(
            static_cast<double>(logits[id] - *max_it)) / denominator);
        selected_sum += selected[i].probability;
    }
    if (normalize_top_k) {
        for (auto& route : selected) route.probability /= selected_sum;
    }
    return selected;
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
    if (options.temperature == 0.0F && options.repetition_penalty == 1.0F) {
        std::size_t best_id = 0;
        float best_logit = logits.front();
        if (!std::isfinite(best_logit)) throw std::runtime_error("model produced non-finite logits");
        for (std::size_t id = 1; id < logits.size(); ++id) {
            const float logit = logits[id];
            if (!std::isfinite(logit)) throw std::runtime_error("model produced non-finite logits");
            if (logit > best_logit) {
                best_logit = logit;
                best_id = id;
            }
        }
        return static_cast<std::uint32_t>(best_id);
    }
    std::vector<float> adjusted;
    std::span<const float> sampling_logits = logits;
    if (options.repetition_penalty != 1.0F) {
        adjusted.assign(logits.begin(), logits.end());
        for (const float logit : adjusted) {
            if (!std::isfinite(logit)) throw std::runtime_error("model produced non-finite logits");
        }
        std::vector<bool> seen(adjusted.size(), false);
        for (const auto id : history) {
            if (id >= adjusted.size()) continue;
            if (seen[id]) continue;
            seen[id] = true;
            adjusted[id] = adjusted[id] < 0.0F
                ? adjusted[id] * options.repetition_penalty
                : adjusted[id] / options.repetition_penalty;
        }
        sampling_logits = adjusted;
    } else {
        for (const float logit : logits) {
            if (!std::isfinite(logit)) throw std::runtime_error("model produced non-finite logits");
        }
    }
    if (options.temperature == 0.0F) {
        return static_cast<std::uint32_t>(std::distance(sampling_logits.begin(),
            std::max_element(sampling_logits.begin(), sampling_logits.end())));
    }

    const double inverse_temperature = 1.0 / static_cast<double>(options.temperature);
    if (options.top_p == 1.0F &&
        (options.top_k == 0 || options.top_k >= sampling_logits.size())) {
        double max_logit = -std::numeric_limits<double>::infinity();
        for (const float logit : sampling_logits) {
            max_logit = std::max(max_logit, static_cast<double>(logit) * inverse_temperature);
        }
        std::vector<double> weights;
        weights.reserve(sampling_logits.size());
        for (const float logit : sampling_logits) {
            weights.push_back(std::exp(static_cast<double>(logit) * inverse_temperature - max_logit));
        }
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        return static_cast<std::uint32_t>(distribution(random));
    }

    std::vector<std::pair<double, std::uint32_t>> ranked;
    ranked.reserve(sampling_logits.size());
    for (std::size_t id = 0; id < sampling_logits.size(); ++id) {
        ranked.emplace_back(static_cast<double>(sampling_logits[id]) * inverse_temperature,
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
    struct DenseBlockReaders {
        DenseBlockReaders(const std::shared_ptr<storage::ModelFile>& file,
                          const Qwen3MoeBlockTensors& block)
            : query(file, block.query), key(file, block.key), value(file, block.value),
              attention_output(file, block.attention_output), router(file, block.router) {}
        TensorReader query;
        TensorReader key;
        TensorReader value;
        TensorReader attention_output;
        TensorReader router;
    };

    Impl(std::filesystem::path path, std::size_t cache_budget, std::size_t requested_context,
         KvCachePrecision kv_precision)
        : model_path(std::move(path)), gguf(model_path),
          file(std::make_shared<storage::ModelFile>(model_path)), index(gguf), tokenizer(gguf),
          embedding_reader(file, index.token_embedding()), output_reader(file, index.output()) {
        const auto& config = index.config();
        model_context = as_size(config.context_length, "context_length");
        // Keep KV residency bounded for mobile devices. The ring retains the most
        // recent tokens while positions continue to advance across the full model context.
        context_capacity = requested_context == 0 ? std::min<std::size_t>(model_context, 512) : requested_context;
        if (context_capacity == 0 || context_capacity > model_context) {
            throw std::invalid_argument("requested KV window exceeds Qwen3-MoE model context");
        }
        const auto layer_count = as_size(config.block_count, "block_count");
        const auto key_dim = as_size(config.key_length, "key_length");
        const auto value_dim = as_size(config.value_length, "value_length");
        const auto kv_heads = as_size(config.key_value_heads, "key_value_heads");
        const auto hidden_size = as_size(config.embedding_length, "embedding_length");
        const auto query_heads = as_size(config.attention_heads, "attention_heads");
        const auto intermediate_size = as_size(config.expert_feed_forward_length,
                                                "expert_feed_forward_length");
        const auto expert_count = as_size(config.expert_count, "expert_count");
        scratch.hidden.resize(hidden_size);
        scratch.embedding_bytes.resize(embedding_reader.row_bytes());
        scratch.normalized.resize(hidden_size);
        scratch.query.resize(query_heads * key_dim);
        scratch.key.resize(kv_heads * key_dim);
        scratch.value.resize(kv_heads * value_dim);
        scratch.attended.resize(query_heads * value_dim);
        scratch.attention_output.resize(hidden_size);
        scratch.router_logits.resize(expert_count);
        scratch.moe_result.resize(hidden_size);
        scratch.gate.resize(intermediate_size);
        scratch.up.resize(intermediate_size);
        scratch.activated.resize(intermediate_size);
        scratch.expert_output.resize(hidden_size);
        scratch.final_hidden.resize(hidden_size);
        if (layer_count > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("model has too many layers for expert cache accounting");
        }
        std::vector<std::size_t> expert_slice_sizes;
        expert_slices.reserve(layer_count);
        expert_slice_sizes.reserve(layer_count * 3);
        for (const auto& block : index.blocks()) {
            std::array<std::vector<storage::ExpertSlice>, 3> slices{
                storage::split_expert_tensor(block.expert_gate, expert_count),
                storage::split_expert_tensor(block.expert_up, expert_count),
                storage::split_expert_tensor(block.expert_down, expert_count)};
            for (const auto& tensor_slices : slices) {
                if (tensor_slices.empty()) throw std::runtime_error("expert tensor has no slices");
                expert_slice_sizes.push_back(tensor_slices.front().size);
            }
            expert_slices.push_back(std::move(slices));
        }
        const auto cache_capacities = allocate_expert_cache_budgets(expert_slice_sizes,
                                                                     cache_budget);
        attention_norms.reserve(layer_count);
        query_norms.reserve(layer_count);
        key_norms.reserve(layer_count);
        feed_forward_norms.reserve(layer_count);
        kv_caches.reserve(layer_count);
        block_readers.reserve(layer_count);
        gate_stores.reserve(layer_count);
        up_stores.reserve(layer_count);
        down_stores.reserve(layer_count);
        for (std::size_t layer = 0; layer < index.blocks().size(); ++layer) {
            const auto& block = index.blocks()[layer];
            attention_norms.push_back(load_vector(file, block.attention_norm));
            query_norms.push_back(load_vector(file, block.query_norm));
            key_norms.push_back(load_vector(file, block.key_norm));
            feed_forward_norms.push_back(load_vector(file, block.feed_forward_norm));
            kv_caches.emplace_back(context_capacity, kv_heads, key_dim, value_dim,
                                   kv_precision);
            block_readers.emplace_back(file, block);
            gate_stores.push_back(std::make_unique<storage::ExpertStore>(
                file, expert_slices[layer][0], cache_capacities[layer * 3]));
            up_stores.push_back(std::make_unique<storage::ExpertStore>(
                file, expert_slices[layer][1], cache_capacities[layer * 3 + 1]));
            down_stores.push_back(std::make_unique<storage::ExpertStore>(
                file, expert_slices[layer][2], cache_capacities[layer * 3 + 2]));
        }
        output_norm = load_vector(file, index.output_norm());
        next_position = 0;
    }

    std::filesystem::path model_path;
    GgufReader gguf;
    std::shared_ptr<storage::ModelFile> file;
    Qwen3MoeIndex index;
    QwenBpeTokenizer tokenizer;
    TensorReader embedding_reader;
    TensorReader output_reader;
    std::size_t model_context{};
    std::size_t context_capacity{};
    std::uint64_t next_position{};
    std::vector<float> output_norm;
    std::vector<std::vector<float>> attention_norms;
    std::vector<std::vector<float>> query_norms;
    std::vector<std::vector<float>> key_norms;
    std::vector<std::vector<float>> feed_forward_norms;
    std::vector<KvCache> kv_caches;
    std::vector<DenseBlockReaders> block_readers;
    std::vector<std::array<std::vector<storage::ExpertSlice>, 3>> expert_slices;
    std::vector<std::unique_ptr<storage::ExpertStore>> gate_stores;
    std::vector<std::unique_ptr<storage::ExpertStore>> up_stores;
    std::vector<std::unique_ptr<storage::ExpertStore>> down_stores;
    struct ForwardScratch {
        std::vector<float> hidden;
        std::vector<std::byte> embedding_bytes;
        std::vector<float> normalized;
        std::vector<float> query;
        std::vector<float> key;
        std::vector<float> value;
        std::vector<float> attended;
        std::vector<float> attention_output;
        std::vector<float> router_logits;
        std::vector<float> moe_result;
        std::vector<float> gate;
        std::vector<float> up;
        std::vector<float> activated;
        std::vector<float> expert_output;
        std::vector<float> final_hidden;
        TensorLinearScratch tensor_linear;
    } scratch;
    struct AssistantTokenSequence {
        std::size_t message_index{};
        std::string text;
        std::vector<std::uint32_t> tokens;
    };
    std::vector<std::uint32_t> session_tokens;
    std::vector<float> session_logits;
    std::vector<AssistantTokenSequence> assistant_token_sequences;
    ExpertLoader expert_loader;

    std::vector<float> forward(std::uint32_t token_id, std::uint64_t position,
                               bool calculate_logits = true) {
        const auto& config = index.config();
        if (token_id >= index.vocabulary_size()) throw std::out_of_range("token id exceeds model vocabulary");
        if (position != next_position || position >= model_context) {
            throw std::invalid_argument("token position is not the next position in this decode state");
        }
        try {
            auto& hidden = scratch.hidden;
            if (index.token_embedding().type == 0 || index.token_embedding().type == 1 ||
                index.token_embedding().type == 30) {
                embedding_reader.read_float_rows_into(token_id, 1, hidden,
                                                       scratch.embedding_bytes);
            } else {
                embedding_reader.read_rows_into(token_id, 1, scratch.embedding_bytes);
                dequantize_quantized_row(index.token_embedding().type,
                                         scratch.embedding_bytes, hidden);
            }
            const auto hidden_size = hidden.size();
            const auto query_heads = as_size(config.attention_heads, "attention_heads");
            const auto kv_heads = as_size(config.key_value_heads, "key_value_heads");
            const auto key_dim = as_size(config.key_length, "key_length");
            const auto rotary_dim = as_size(config.rotary_dimension, "rotary_dimension");
            const auto epsilon = static_cast<float>(config.rms_norm_epsilon);
            const auto theta = static_cast<float>(config.rope_frequency_base);
            for (std::size_t layer = 0; layer < index.blocks().size(); ++layer) {
                const auto& block = index.blocks()[layer];
                const auto& readers = block_readers[layer];
                auto& normalized = scratch.normalized;
                rms_norm(hidden, attention_norms[layer], normalized, epsilon);
                auto& query = scratch.query;
                auto& key = scratch.key;
                auto& value = scratch.value;
                tensor_linear(readers.query, normalized, query, 0, &scratch.tensor_linear);
                tensor_linear(readers.key, normalized, key, 0, &scratch.tensor_linear);
                tensor_linear(readers.value, normalized, value, 0, &scratch.tensor_linear);
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
                auto& attended = scratch.attended;
                kv_caches[layer].attend(query.data(), query.size(), attended.data(), attended.size());
                auto& attention_output = scratch.attention_output;
                tensor_linear(readers.attention_output, attended, attention_output,
                              0, &scratch.tensor_linear);
                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += attention_output[i];

                rms_norm(hidden, feed_forward_norms[layer], normalized, epsilon);
                auto& router_logits = scratch.router_logits;
                tensor_linear(readers.router, normalized, router_logits,
                              0, &scratch.tensor_linear);
                const auto routes = route_experts(router_logits,
                    as_size(config.experts_per_token, "experts_per_token"), config.expert_weights_norm);
                auto& moe_result = scratch.moe_result;
                std::fill(moe_result.begin(), moe_result.end(), 0.0F);
                const auto& gate_slices = expert_slices[layer][0];
                const auto& up_slices = expert_slices[layer][1];
                const auto& down_slices = expert_slices[layer][2];
                const auto load_expert = [this, layer](std::size_t expert_id) {
                    auto up = expert_loader.submit([this, layer, expert_id] {
                        return up_stores[layer]->get(expert_id);
                    });
                    auto down = expert_loader.submit([this, layer, expert_id] {
                        return down_stores[layer]->get(expert_id);
                    });
                    auto gate = gate_stores[layer]->get(expert_id);
                    return ExpertWeights{std::move(gate), up.get(), down.get()};
                };
                auto pending_weights = expert_loader.submit(
                    [load_expert, expert_id = routes.front().id] { return load_expert(expert_id); });
                for (std::size_t route_index = 0; route_index < routes.size(); ++route_index) {
                    const auto& route = routes[route_index];
                    auto weights = pending_weights.get();
                    if (route_index + 1 < routes.size()) {
                        pending_weights = expert_loader.submit(
                            [load_expert, expert_id = routes[route_index + 1].id] {
                                return load_expert(expert_id);
                            });
                    }
                    TensorReader gate_reader(expert_view(block.expert_gate, gate_slices[route.id]), weights.gate);
                    TensorReader up_reader(expert_view(block.expert_up, up_slices[route.id]), weights.up);
                    TensorReader down_reader(expert_view(block.expert_down, down_slices[route.id]), weights.down);
                    auto& gate = scratch.gate;
                    auto& up = scratch.up;
                    auto& activated = scratch.activated;
                    tensor_linear(gate_reader, normalized, gate, 0, &scratch.tensor_linear);
                    tensor_linear(up_reader, normalized, up, 0, &scratch.tensor_linear);
                    for (std::size_t i = 0; i < activated.size(); ++i) {
                        activated[i] = (gate[i] / (1.0F + std::exp(-gate[i]))) * up[i];
                    }
                    auto& expert_output = scratch.expert_output;
                    tensor_linear(down_reader, activated, expert_output,
                                  0, &scratch.tensor_linear);
                    for (std::size_t i = 0; i < hidden_size; ++i) {
                        moe_result[i] += route.probability * expert_output[i];
                    }
                }
                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += moe_result[i];
            }
            ++next_position;
            if (!calculate_logits) return {};
            auto& final_hidden = scratch.final_hidden;
            rms_norm(hidden, output_norm, final_hidden, epsilon);
            std::vector<float> logits(as_size(index.vocabulary_size(), "vocabulary_size"));
            tensor_linear(output_reader, final_hidden, logits, 0, &scratch.tensor_linear);
            return logits;
        } catch (...) {
            reset();
            throw;
        }
    }

    void reset() {
        for (auto& cache : kv_caches) cache.clear();
        next_position = 0;
        session_tokens.clear();
        session_logits.clear();
        assistant_token_sequences.clear();
    }

    ExpertCacheStats cache_stats() const {
        ExpertCacheStats stats;
        const auto accumulate = [&stats](const auto& stores) {
            for (const auto& store : stores) {
                stats.capacity_bytes += store->cache_capacity_bytes();
                stats.resident_bytes += store->cache_bytes();
                stats.bytes_read += store->bytes_read_from_disk();
                stats.read_operations += store->read_operations();
                stats.read_time_ns += store->read_time_ns();
                stats.hits += store->cache_hits();
                stats.misses += store->cache_misses();
            }
        };
        accumulate(gate_stores);
        accumulate(up_stores);
        accumulate(down_stores);
        return stats;
    }

    std::size_t kv_storage_bytes() const noexcept {
        std::size_t total = 0;
        for (const auto& cache : kv_caches) {
            const auto bytes = cache.storage_bytes();
            if (bytes > std::numeric_limits<std::size_t>::max() - total) {
                return std::numeric_limits<std::size_t>::max();
            }
            total += bytes;
        }
        return total;
    }

    std::vector<std::uint32_t> generate(std::span<const std::uint32_t> prompt,
                                        std::size_t max_new_tokens,
                                        const GenerationOptions& options,
                                        GenerationStats* stats = nullptr,
                                        bool reuse_session_prefix = false) {
        if (stats) *stats = {};
        if (prompt.empty()) throw std::invalid_argument("generation prompt must contain at least one token");
        if (prompt.size() > model_context || max_new_tokens > model_context - prompt.size()) {
            throw std::length_error("prompt and generation length exceed model context");
        }
        const bool prefix_matches = reuse_session_prefix &&
            session_tokens.size() <= prompt.size() &&
            std::equal(session_tokens.begin(), session_tokens.end(), prompt.begin()) &&
            next_position == session_tokens.size() &&
            (session_tokens.size() < prompt.size() || !session_logits.empty());
        if (!prefix_matches) {
            if (reuse_session_prefix) {
                auto retained_assistant_tokens = std::move(assistant_token_sequences);
                reset();
                assistant_token_sequences = std::move(retained_assistant_tokens);
            } else {
                reset();
            }
        }
        std::vector<std::uint32_t> generated;
        if (max_new_tokens == 0) return generated;
        std::vector<std::uint32_t> history;
        if (options.repetition_penalty != 1.0F) {
            history.assign(prompt.begin(), prompt.end());
            history.reserve(prompt.size() + max_new_tokens);
        }
        std::vector<float> logits = prefix_matches ? session_logits : std::vector<float>{};
        try {
            const auto prefill_start = std::chrono::steady_clock::now();
            const auto prefill_start_position = session_tokens.size();
            for (std::size_t i = prefill_start_position; i < prompt.size(); ++i) {
                logits = forward(prompt[i], i, i + 1 == prompt.size());
                session_tokens.push_back(prompt[i]);
            }
            if (stats) {
                stats->prefill_time_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - prefill_start).count());
            }
            const auto decode_start = std::chrono::steady_clock::now();
            std::mt19937_64 random(options.seed);
            generated.reserve(max_new_tokens);
            for (std::size_t i = 0; i < max_new_tokens; ++i) {
                const auto token = sample_token(logits, history, options, random);
                generated.push_back(token);
                if (options.repetition_penalty != 1.0F) history.push_back(token);
                if (token == tokenizer.eos_token_id()) break;
                if (i + 1 < max_new_tokens) {
                    logits = forward(token, prompt.size() + i);
                    session_tokens.push_back(token);
                }
            }
            if (stats) {
                stats->decode_time_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - decode_start).count());
                stats->generated_tokens = generated.size();
            }
            session_logits = std::move(logits);
        } catch (...) {
            reset();
            throw;
        }
        return generated;
    }
};

Qwen3MoeRunner::Qwen3MoeRunner(std::filesystem::path model_path,
                               std::size_t expert_cache_budget_bytes,
                               std::size_t context_capacity,
                               KvCachePrecision kv_precision) {
    impl_ = std::make_unique<Impl>(std::move(model_path), expert_cache_budget_bytes,
                                   context_capacity, kv_precision);
}

Qwen3MoeRunner::~Qwen3MoeRunner() = default;
Qwen3MoeRunner::Qwen3MoeRunner(Qwen3MoeRunner&&) noexcept = default;
Qwen3MoeRunner& Qwen3MoeRunner::operator=(Qwen3MoeRunner&&) noexcept = default;

std::vector<float> Qwen3MoeRunner::forward_token(std::uint32_t token_id, std::uint64_t position) {
    if (!impl_) throw std::logic_error("Qwen3-MoE runner has been moved from");
    return impl_->forward(token_id, position, true);
}
std::vector<std::uint32_t> Qwen3MoeRunner::generate_tokens(
    std::span<const std::uint32_t> prompt, std::size_t max_new_tokens,
    const GenerationOptions& options, GenerationStats* stats) {
    if (!impl_) throw std::logic_error("Qwen3-MoE runner has been moved from");
    return impl_->generate(prompt, max_new_tokens, options, stats);
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
std::string Qwen3MoeRunner::generate_chat(std::string_view user_message,
                                          std::size_t max_new_tokens,
                                          const GenerationOptions& options) {
    const ChatMessage message{"user", user_message};
    return generate_chat(std::span<const ChatMessage>(&message, 1), max_new_tokens, options);
}
std::string Qwen3MoeRunner::generate_chat(std::span<const ChatMessage> messages,
                                          std::size_t max_new_tokens,
                                          const GenerationOptions& options) {
    if (!impl_) throw std::logic_error("Qwen3-MoE runner has been moved from");
    if (messages.empty()) throw std::invalid_argument("chat history must contain at least one message");
    std::vector<std::uint32_t> prompt_tokens;
    std::string_view previous_role;
    std::size_t content_bytes = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        const bool is_system = message.role == "system";
        const bool is_user = message.role == "user";
        const bool is_assistant = message.role == "assistant";
        if (!is_system && !is_user && !is_assistant) {
            throw std::invalid_argument("chat role must be system, user, or assistant");
        }
        if (i == 0) {
            if (!is_system && !is_user) {
                throw std::invalid_argument("chat history must begin with a system or user message");
            }
        } else if (is_system) {
            throw std::invalid_argument("system message is only allowed at the start of chat history");
        } else if (previous_role == "system") {
            if (!is_user) throw std::invalid_argument("a system message must be followed by a user message");
        } else if ((previous_role == "user" && !is_assistant) ||
                   (previous_role == "assistant" && !is_user)) {
            throw std::invalid_argument("chat user and assistant messages must alternate");
        }
        if (message.content.size() > std::numeric_limits<std::size_t>::max() - content_bytes) {
            throw std::length_error("chat history is too large");
        }
        content_bytes += message.content.size();
        previous_role = message.role;
    }
    if (previous_role != "user") {
        throw std::invalid_argument("chat history must end with a user message");
    }
    constexpr std::size_t message_overhead = 32;
    if (messages.size() > (std::numeric_limits<std::size_t>::max() - 24) / message_overhead) {
        throw std::length_error("chat history is too large");
    }
    const auto prompt_overhead = messages.size() * message_overhead + 24;
    if (content_bytes > std::numeric_limits<std::size_t>::max() - prompt_overhead) {
        throw std::length_error("chat history is too large");
    }
    prompt_tokens.reserve(content_bytes + prompt_overhead);
    const auto append_encoded = [this, &prompt_tokens](std::string_view text) {
        auto encoded = impl_->tokenizer.encode(std::string(text));
        prompt_tokens.insert(prompt_tokens.end(), encoded.begin(), encoded.end());
    };
    for (std::size_t message_index = 0; message_index < messages.size(); ++message_index) {
        const auto& message = messages[message_index];
        std::string header = "<|im_start|>";
        header.append(message.role);
        header.push_back('\n');
        const auto assistant_tokens = message.role == "assistant"
            ? std::find_if(impl_->assistant_token_sequences.rbegin(),
                           impl_->assistant_token_sequences.rend(),
                [message_index, &message](
                    const Impl::AssistantTokenSequence& sequence) {
                    return sequence.message_index == message_index && sequence.text == message.content;
                })
            : impl_->assistant_token_sequences.rend();
        if (assistant_tokens != impl_->assistant_token_sequences.rend()) {
            append_encoded(header);
            prompt_tokens.insert(prompt_tokens.end(), assistant_tokens->tokens.begin(),
                                 assistant_tokens->tokens.end());
        } else {
            header.append(message.content);
            append_encoded(header);
        }
        append_encoded("<|im_end|>\n");
    }
    append_encoded("<|im_start|>assistant\n");
    const auto generated = impl_->generate(prompt_tokens, max_new_tokens, options,
                                           nullptr, true);
    std::vector<std::uint32_t> printable;
    printable.reserve(generated.size());
    for (const auto token : generated) {
        if (token == impl_->tokenizer.eos_token_id()) break;
        printable.push_back(token);
    }
    auto response = impl_->tokenizer.decode(printable);
    const auto reply_index = messages.size();
    std::erase_if(impl_->assistant_token_sequences, [reply_index](const auto& sequence) {
        return sequence.message_index == reply_index;
    });
    impl_->assistant_token_sequences.push_back({reply_index, response, std::move(printable)});
    return response;
}
void Qwen3MoeRunner::reset() { if (impl_) impl_->reset(); }
const TransformerConfig& Qwen3MoeRunner::config() const noexcept {
    static const TransformerConfig empty{};
    return impl_ ? impl_->index.config() : empty;
}
std::size_t Qwen3MoeRunner::kv_cache_storage_bytes() const noexcept {
    return impl_ ? impl_->kv_storage_bytes() : 0;
}
std::uint64_t Qwen3MoeRunner::bytes_read_from_disk() const noexcept {
    return impl_ && impl_->file ? impl_->file->bytes_read() : 0;
}
ExpertCacheStats Qwen3MoeRunner::expert_cache_stats() const {
    return impl_ ? impl_->cache_stats() : ExpertCacheStats{};
}

}  // namespace pokitlms::model
