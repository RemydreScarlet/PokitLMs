#include "pokitlms/model/qwen35_runner.hpp"

#include "pokitlms/model/qwen_bpe_tokenizer.hpp"
#include "pokitlms/model/tensor_linear.hpp"
#include "pokitlms/model/tensor_reader.hpp"
#include "pokitlms/ops/causal_conv1d.hpp"
#include "pokitlms/ops/gated_delta_net.hpp"
#include "pokitlms/ops/quantized_linear.hpp"
#include "pokitlms/ops/rms_norm.hpp"
#include "pokitlms/storage/expert_store.hpp"
#include "simd_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace pokitlms::model {
namespace {

std::size_t as_size(std::uint64_t value, const char* label) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("Qwen3.5 dimension exceeds address space: ") + label);
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* label) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::invalid_argument(std::string("Qwen3.5 dimension overflow: ") + label);
    }
    return lhs * rhs;
}

std::vector<float> load_vector(const std::shared_ptr<storage::ModelFile>& file,
                               const TensorInfo& tensor) {
    if (tensor.dimensions.size() != 1) {
        throw std::invalid_argument("expected a one-dimensional Qwen3.5 weight: " + tensor.name);
    }
    TensorReader reader(file, tensor);
    return reader.read_float_rows(0, 1);
}

std::vector<float> load_matrix_values(const std::shared_ptr<storage::ModelFile>& file,
                                      const TensorInfo& tensor) {
    TensorReader reader(file, tensor);
    if (reader.row_count() > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("Qwen3.5 tensor has too many rows: " + tensor.name);
    }
    return reader.read_float_rows(0, static_cast<std::size_t>(reader.row_count()));
}

void prepare_rope(std::uint64_t position, std::size_t rotary_dimension,
                  const std::array<std::uint64_t, 4>& sections, float theta,
                  std::span<float> cosine, std::span<float> sine) {
    if (rotary_dimension == 0 || rotary_dimension % 2 != 0 ||
        cosine.size() != rotary_dimension / 2 || sine.size() != cosine.size() ||
        !std::isfinite(theta) || theta <= 0.0F) {
        throw std::invalid_argument("invalid Qwen3.5 RoPE parameters");
    }
    const auto half = rotary_dimension / 2;
    const auto section_total = std::accumulate(sections.begin(), sections.end(),
                                               std::uint64_t{});
    if (section_total == 0) throw std::invalid_argument("empty Qwen3.5 MRoPE sections");
    // Text input uses [position, position, position, 0] for the four MRoPE
    // axes. Multimodal image/video positions require a separate path.
    for (std::size_t pair = 0; pair < half; ++pair) {
        const float exponent = -static_cast<float>(pair) / static_cast<float>(half);
        const auto sector = static_cast<std::uint64_t>(pair) % section_total;
        const bool extra_axis = sector >= sections[0] + sections[1] + sections[2];
        const auto axis_position = extra_axis ? 0.0F : static_cast<float>(position);
        const float angle = axis_position * std::pow(theta, exponent);
        cosine[pair] = std::cos(angle);
        sine[pair] = std::sin(angle);
    }
}

void apply_rope(std::span<float> values, std::size_t head_dimension,
                std::size_t rotary_dimension, std::span<const float> cosine,
                std::span<const float> sine) {
    if (head_dimension == 0 || values.size() % head_dimension != 0 ||
        rotary_dimension == 0 || rotary_dimension > head_dimension ||
        rotary_dimension % 2 != 0 || cosine.size() != rotary_dimension / 2 ||
        sine.size() != cosine.size()) {
        throw std::invalid_argument("invalid Qwen3.5 RoPE dimensions");
    }
    const auto heads = values.size() / head_dimension;
    const auto half = rotary_dimension / 2;
    for (std::size_t head = 0; head < heads; ++head) {
        auto* vector = values.data() + head * head_dimension;
        detail::rope_rotate_f32(vector, vector + half, cosine.data(), sine.data(), half);
    }
}

float sigmoid(float value) noexcept {
    if (value >= 0.0F) return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float softplus(float value) noexcept {
    if (value > 20.0F) return value;
    if (value < -20.0F) return std::exp(value);
    return std::log1p(std::exp(value));
}

float silu(float value) noexcept { return value * sigmoid(value); }

struct ExpertRoute {
    std::size_t id{};
    float weight{};
    float logit{};
};

std::vector<ExpertRoute> select_experts(std::span<const float> logits, std::size_t top_k,
                                        bool normalize) {
    if (top_k == 0 || top_k > logits.size()) throw std::invalid_argument("invalid Qwen3.5 expert top-k");
    const float maximum = *std::max_element(logits.begin(), logits.end());
    if (!std::isfinite(maximum) || std::any_of(logits.begin(), logits.end(),
        [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("Qwen3.5 expert router produced non-finite logits");
    }
    float denominator = 0.0F;
    std::vector<float> probabilities(logits.size());
    for (std::size_t id = 0; id < logits.size(); ++id) {
        probabilities[id] = std::exp(logits[id] - maximum);
        denominator += probabilities[id];
    }
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
        throw std::runtime_error("Qwen3.5 expert router softmax is invalid");
    }
    for (auto& probability : probabilities) probability /= denominator;

    std::vector<ExpertRoute> routes;
    routes.reserve(top_k);
    for (std::size_t id = 0; id < logits.size(); ++id) {
        const ExpertRoute candidate{id, probabilities[id], logits[id]};
        const auto where = std::lower_bound(routes.begin(), routes.end(), candidate,
            [](const ExpertRoute& lhs, const ExpertRoute& rhs) {
                return lhs.logit > rhs.logit || (lhs.logit == rhs.logit && lhs.id < rhs.id);
            });
        if (routes.size() < top_k) routes.insert(where, candidate);
        else if (where != routes.end()) {
            routes.insert(where, candidate);
            routes.pop_back();
        }
    }
    if (normalize) {
        float sum = 0.0F;
        for (const auto& route : routes) sum += route.weight;
        if (!std::isfinite(sum) || sum <= 0.0F) {
            throw std::runtime_error("Qwen3.5 selected expert weights are invalid");
        }
        for (auto& route : routes) route.weight /= sum;
    }
    return routes;
}

std::size_t automatic_expert_cache_budget() {
    constexpr std::size_t fallback = 128U * 1024U * 1024U;
#if defined(__ANDROID__)
    constexpr std::uint64_t ceiling = 2ULL * 1024U * 1024U * 1024U;
#else
    constexpr std::uint64_t ceiling = 4ULL * 1024U * 1024U * 1024U;
#endif
    std::uint64_t available = 0;
#if defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemAvailable:" && unit == "kB") {
            available = value > std::numeric_limits<std::uint64_t>::max() / 1024U
                ? std::numeric_limits<std::uint64_t>::max() : value * 1024U;
            break;
        }
    }
#endif
#if !defined(_WIN32) && defined(_SC_AVPHYS_PAGES)
    if (available == 0) {
        const auto pages = ::sysconf(_SC_AVPHYS_PAGES);
        const auto page_size = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0 &&
            static_cast<std::uint64_t>(pages) <=
                std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(page_size)) {
            available = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
        }
    }
#endif
    if (available == 0) return fallback;
    return static_cast<std::size_t>(std::min<std::uint64_t>(available / 4U, ceiling));
}

TensorInfo expert_view(const TensorInfo& tensor, const storage::ExpertSlice& slice) {
    TensorInfo view;
    view.name = tensor.name;
    view.dimensions = {tensor.dimensions[0], tensor.dimensions[1]};
    view.type = tensor.type;
    view.file_offset = slice.offset;
    view.payload_size = slice.size;
    return view;
}

class ExpertIoLoader {
public:
    explicit ExpertIoLoader(std::size_t worker_count) {
        workers_.reserve(worker_count);
        try {
            for (std::size_t i = 0; i < worker_count; ++i) {
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

    ~ExpertIoLoader() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    ExpertIoLoader(const ExpertIoLoader&) = delete;
    ExpertIoLoader& operator=(const ExpertIoLoader&) = delete;

    template <typename Work>
    [[nodiscard]] auto submit(Work&& work) {
        using Result = std::invoke_result_t<Work>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Work>(work));
        auto result = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) throw std::logic_error("Qwen3.5 expert loader is stopping");
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

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> tasks_;
    bool stopping_{};
    std::vector<std::thread> workers_;
};

struct PendingExpertWeights {
    std::future<std::shared_ptr<const std::vector<std::byte>>> gate;
    std::future<std::shared_ptr<const std::vector<std::byte>>> up;
    std::future<std::shared_ptr<const std::vector<std::byte>>> down;
};

struct LayerReaders {
    explicit LayerReaders(const std::shared_ptr<storage::ModelFile>& file,
                          const Qwen35LayerTensors& tensors, bool mixture_of_experts)
        : full_attention(tensors.full_attention),
          attention_norm(load_vector(file, tensors.attention_norm)),
          post_attention_norm(load_vector(file, tensors.post_attention_norm)) {
        if (mixture_of_experts) {
            router = std::make_unique<TensorReader>(file, tensors.router);
            shared_expert_router = std::make_unique<TensorReader>(file, tensors.shared_expert_router);
            shared_expert_gate = std::make_unique<TensorReader>(file, tensors.shared_expert_gate);
            shared_expert_up = std::make_unique<TensorReader>(file, tensors.shared_expert_up);
            shared_expert_down = std::make_unique<TensorReader>(file, tensors.shared_expert_down);
        } else {
            feed_forward_gate = std::make_unique<TensorReader>(file, tensors.feed_forward_gate);
            feed_forward_up = std::make_unique<TensorReader>(file, tensors.feed_forward_up);
            feed_forward_down = std::make_unique<TensorReader>(file, tensors.feed_forward_down);
        }
        if (full_attention) {
            query = std::make_unique<TensorReader>(file, tensors.query);
            key = std::make_unique<TensorReader>(file, tensors.key);
            value = std::make_unique<TensorReader>(file, tensors.value);
            query_norm = load_vector(file, tensors.query_norm);
            key_norm = load_vector(file, tensors.key_norm);
            attention_output = std::make_unique<TensorReader>(file, tensors.attention_output);
        } else {
            linear_qkv = std::make_unique<TensorReader>(file, tensors.linear_qkv);
            linear_gate = std::make_unique<TensorReader>(file, tensors.linear_gate);
            linear_alpha = std::make_unique<TensorReader>(file, tensors.linear_alpha);
            linear_beta = std::make_unique<TensorReader>(file, tensors.linear_beta);
            linear_dt_bias = load_vector(file, tensors.linear_dt_bias);
            linear_a = load_vector(file, tensors.linear_a);
            linear_norm = load_vector(file, tensors.linear_norm);
            linear_convolution = load_matrix_values(file, tensors.linear_convolution);
            linear_output = std::make_unique<TensorReader>(file, tensors.linear_output);
        }
    }

    bool full_attention;
    std::unique_ptr<TensorReader> feed_forward_gate;
    std::unique_ptr<TensorReader> feed_forward_up;
    std::unique_ptr<TensorReader> feed_forward_down;
    std::unique_ptr<TensorReader> router;
    std::unique_ptr<TensorReader> shared_expert_router;
    std::unique_ptr<TensorReader> shared_expert_gate;
    std::unique_ptr<TensorReader> shared_expert_up;
    std::unique_ptr<TensorReader> shared_expert_down;
    std::vector<float> attention_norm;
    std::vector<float> post_attention_norm;
    std::unique_ptr<TensorReader> query;
    std::unique_ptr<TensorReader> key;
    std::unique_ptr<TensorReader> value;
    std::unique_ptr<TensorReader> attention_output;
    std::unique_ptr<TensorReader> linear_qkv;
    std::unique_ptr<TensorReader> linear_gate;
    std::unique_ptr<TensorReader> linear_alpha;
    std::unique_ptr<TensorReader> linear_beta;
    std::unique_ptr<TensorReader> linear_output;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
    std::vector<float> linear_dt_bias;
    std::vector<float> linear_a;
    std::vector<float> linear_norm;
    std::vector<float> linear_convolution;
};

struct LinearState {
    std::vector<float> convolution;
    std::vector<float> recurrent;
    GatedDeltaNetScratch scratch;
};

struct ForwardScratch {
    std::vector<float> hidden;
    std::vector<float> normalized;
    std::vector<float> projected;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> gate;
    std::vector<float> attended;
    std::vector<float> mixer_output;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_activated;
    std::vector<float> router_logits;
    std::vector<float> moe_result;
    std::vector<float> expert_output;
    std::vector<float> final_hidden;
    std::vector<float> rope_cosine;
    std::vector<float> rope_sine;
    std::vector<std::byte> embedding;
    TensorLinearScratch linear;
};

}  // namespace

class Qwen35Runner::Impl {
public:
    struct RoutedExpertStores {
        std::array<std::vector<storage::ExpertSlice>, 3> slices;
        std::unique_ptr<storage::ExpertStore> gate;
        std::unique_ptr<storage::ExpertStore> up;
        std::unique_ptr<storage::ExpertStore> down;
    };

    Impl(std::filesystem::path path, std::size_t requested_context,
         KvCachePrecision kv_precision, std::size_t expert_io_threads)
        : model_path(std::move(path)), gguf(model_path),
          file(std::make_shared<storage::ModelFile>(model_path)), index(gguf), tokenizer(gguf),
          embedding_reader(file, index.token_embedding()), output_reader(file, index.output()),
          output_norm(load_vector(file, index.output_norm())) {
        const auto& config = index.config();
        if (expert_io_threads == 0 || expert_io_threads > 4) {
            throw std::invalid_argument("Qwen3.5 expert I/O threads must be between 1 and 4");
        }
        model_context = as_size(config.context_length, "context_length");
        context_capacity = requested_context == 0 ? std::min<std::size_t>(model_context, 512)
                                                   : requested_context;
        if (context_capacity == 0 || context_capacity > model_context) {
            throw std::invalid_argument("requested KV window exceeds Qwen3.5 model context");
        }

        const auto hidden = as_size(config.embedding_length, "embedding_length");
        const auto ffn = as_size(config.mixture_of_experts
            ? std::max(config.expert_feed_forward_length,
                       config.shared_expert_feed_forward_length)
            : config.feed_forward_length, "feed_forward_length");
        const auto attention_heads = as_size(config.attention_heads, "attention_heads");
        const auto kv_heads = as_size(config.key_value_heads, "key_value_heads");
        const auto head_dim = as_size(config.attention_head_length, "attention_head_length");
        const auto value_dim = as_size(config.value_head_length, "value_head_length");
        const auto rotary_dim = as_size(config.rotary_dimension, "rotary_dimension");
        const auto linear_qk_heads = as_size(config.linear_key_head_count(), "linear_key_heads");
        const auto linear_qk_dim = as_size(config.linear_key_head_dimension(), "linear_key_dimension");
        const auto linear_v_heads = as_size(config.linear_value_head_count(), "linear_value_heads");
        const auto linear_v_dim = as_size(config.linear_value_head_dimension(), "linear_value_dimension");
        const auto conv_kernel = as_size(config.ssm_conv_kernel, "ssm_conv_kernel");
        const auto conv_channels = as_size(config.ssm_inner_size + 2 * config.linear_key_dimension(),
                                           "ssm_conv_channels");
        if (linear_qk_heads == 0 || linear_v_heads % linear_qk_heads != 0 ||
            checked_product(linear_qk_heads, linear_qk_dim, "linear key width") !=
                config.linear_key_dimension() ||
            checked_product(linear_v_heads, linear_v_dim, "linear value width") !=
                config.ssm_inner_size) {
            throw std::invalid_argument("inconsistent Qwen3.5 linear-attention head dimensions");
        }

        scratch.hidden.resize(hidden);
        scratch.normalized.resize(hidden);
        const auto attention_q_width = checked_product(attention_heads,
            checked_product(head_dim, 2, "attention query head"), "attention query projection");
        const auto linear_qk_width = checked_product(linear_qk_heads, linear_qk_dim,
                                                      "linear query/key width");
        const auto linear_v_width = checked_product(linear_v_heads, linear_v_dim,
                                                     "linear value width");
        const auto linear_qkv_width = checked_product(2, linear_qk_width,
                                                       "linear QKV projection") + linear_v_width;
        scratch.projected.resize(std::max({hidden, ffn, attention_q_width, linear_qkv_width}));
        scratch.query.resize(checked_product(attention_heads, head_dim, "query"));
        scratch.key.resize(checked_product(kv_heads, head_dim, "key"));
        scratch.value.resize(checked_product(kv_heads, value_dim, "value"));
        scratch.gate.resize(std::max(checked_product(attention_heads, value_dim, "attention gate"),
                                     checked_product(linear_v_heads, linear_v_dim, "linear gate")));
        scratch.attended.resize(std::max(checked_product(attention_heads, value_dim, "attention output"),
                                         checked_product(linear_v_heads, linear_v_dim, "linear output")));
        scratch.mixer_output.resize(hidden);
        scratch.ffn_gate.resize(ffn);
        scratch.ffn_up.resize(ffn);
        scratch.ffn_activated.resize(ffn);
        if (config.mixture_of_experts) {
            scratch.router_logits.resize(as_size(config.expert_count, "expert_count"));
            scratch.moe_result.resize(hidden);
            scratch.expert_output.resize(hidden);
        }
        scratch.final_hidden.resize(hidden);
        scratch.embedding.resize(embedding_reader.row_bytes());
        scratch.rope_cosine.resize(rotary_dim / 2);
        scratch.rope_sine.resize(rotary_dim / 2);

        layers.reserve(index.layers().size());
        if (config.mixture_of_experts) {
            expert_loader = std::make_unique<ExpertIoLoader>(expert_io_threads);
            const auto layer_count = index.layers().size();
            if (layer_count > std::numeric_limits<std::size_t>::max() / 3U) {
                throw std::invalid_argument("Qwen3.5 expert cache count overflows");
            }
            const auto per_store_budget = automatic_expert_cache_budget() / (layer_count * 3U);
            routed_expert_stores.reserve(layer_count);
            for (const auto& tensors : index.layers()) {
                RoutedExpertStores stores;
                const auto expert_count = as_size(config.expert_count, "expert_count");
                stores.slices = {
                    storage::split_expert_tensor(tensors.expert_gate, expert_count),
                    storage::split_expert_tensor(tensors.expert_up, expert_count),
                    storage::split_expert_tensor(tensors.expert_down, expert_count)};
                stores.gate = std::make_unique<storage::ExpertStore>(file, stores.slices[0],
                                                                     per_store_budget);
                stores.up = std::make_unique<storage::ExpertStore>(file, stores.slices[1],
                                                                   per_store_budget);
                stores.down = std::make_unique<storage::ExpertStore>(file, stores.slices[2],
                                                                     per_store_budget);
                routed_expert_stores.push_back(std::move(stores));
            }
        }
        linear_states.resize(index.layers().size());
        attention_caches.resize(index.layers().size());
        for (std::size_t layer = 0; layer < index.layers().size(); ++layer) {
            const auto& tensors = index.layers()[layer];
            layers.emplace_back(file, tensors, config.mixture_of_experts);
            if (tensors.full_attention) {
                attention_caches[layer] = std::make_unique<KvCache>(context_capacity, kv_heads,
                    head_dim, value_dim, kv_precision, attention_heads);
            } else {
                auto& state = linear_states[layer];
                state.convolution.resize(checked_product(conv_channels, conv_kernel - 1,
                                                          "linear convolution state"));
                state.recurrent.resize(checked_product(linear_v_heads,
                    checked_product(linear_qk_dim, linear_v_dim, "linear state head"),
                    "linear recurrent state"));
            }
        }
    }

    std::filesystem::path model_path;
    GgufReader gguf;
    std::shared_ptr<storage::ModelFile> file;
    Qwen35Index index;
    QwenBpeTokenizer tokenizer;
    TensorReader embedding_reader;
    TensorReader output_reader;
    std::vector<float> output_norm;
    std::vector<LayerReaders> layers;
    std::vector<RoutedExpertStores> routed_expert_stores;
    std::unique_ptr<ExpertIoLoader> expert_loader;
    std::vector<LinearState> linear_states;
    std::vector<std::unique_ptr<KvCache>> attention_caches;
    std::size_t model_context{};
    std::size_t context_capacity{};
    std::uint64_t next_position{};
    std::shared_ptr<TensorLinearBackend> linear_backend;
    ForwardScratch scratch;

    void forward_into(std::uint32_t token_id, std::uint64_t position,
                      bool calculate_logits, std::vector<float>* logits) {
        const auto& config = index.config();
        if (token_id >= index.vocabulary_size()) throw std::out_of_range("token id exceeds model vocabulary");
        if (position != next_position || position >= model_context) {
            throw std::invalid_argument("token position is not next in the Qwen3.5 decode state");
        }
        try {
            if (index.token_embedding().type == 0 || index.token_embedding().type == 1 ||
                index.token_embedding().type == 30) {
                embedding_reader.read_float_rows_into(token_id, 1, scratch.hidden, scratch.embedding);
            } else {
                embedding_reader.read_rows_into(token_id, 1, scratch.embedding);
                dequantize_quantized_row(index.token_embedding().type, scratch.embedding, scratch.hidden);
            }

            const auto hidden_size = scratch.hidden.size();
            const auto attention_heads = as_size(config.attention_heads, "attention_heads");
            const auto kv_heads = as_size(config.key_value_heads, "key_value_heads");
            const auto head_dim = as_size(config.attention_head_length, "attention_head_length");
            const auto value_dim = as_size(config.value_head_length, "value_head_length");
            const auto rotary_dim = as_size(config.rotary_dimension, "rotary_dimension");
            const auto linear_qk_heads = as_size(config.linear_key_head_count(), "linear_key_heads");
            const auto linear_qk_dim = as_size(config.linear_key_head_dimension(), "linear_key_dimension");
            const auto linear_v_heads = as_size(config.linear_value_head_count(), "linear_value_heads");
            const auto linear_v_dim = as_size(config.linear_value_head_dimension(), "linear_value_dimension");
            const auto conv_kernel = as_size(config.ssm_conv_kernel, "ssm_conv_kernel");
            const float epsilon = static_cast<float>(config.rms_norm_epsilon);
            const float theta = static_cast<float>(config.rope_frequency_base);
            prepare_rope(position, rotary_dim, config.rope_dimension_sections, theta,
                         scratch.rope_cosine, scratch.rope_sine);

            for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                auto& readers = layers[layer];
                auto& hidden = scratch.hidden;
                auto& normalized = scratch.normalized;
                rms_norm(hidden, readers.attention_norm, normalized, epsilon);
                if (readers.full_attention) {
                    const auto query_width = checked_product(attention_heads,
                                                              checked_product(head_dim, 2, "gated query head"),
                                                              "gated query");
                    tensor_linear(*readers.query, normalized,
                        std::span<float>(scratch.projected.data(), query_width), 0, &scratch.linear);
                    auto& query = scratch.query;
                    auto& gate = scratch.gate;
                    for (std::size_t head = 0; head < attention_heads; ++head) {
                        const auto in = head * head_dim * 2;
                        const auto out = head * head_dim;
                        std::copy_n(scratch.projected.data() + in, head_dim, query.data() + out);
                        std::copy_n(scratch.projected.data() + in + head_dim, head_dim,
                                    gate.data() + out);
                    }
                    tensor_linear(*readers.key, normalized, scratch.key, 0, &scratch.linear);
                    tensor_linear(*readers.value, normalized, scratch.value, 0, &scratch.linear);
                    for (std::size_t head = 0; head < attention_heads; ++head) {
                        auto row = std::span<float>(query).subspan(head * head_dim, head_dim);
                        rms_norm(row, readers.query_norm, row, epsilon);
                    }
                    for (std::size_t head = 0; head < kv_heads; ++head) {
                        auto row = std::span<float>(scratch.key).subspan(head * head_dim, head_dim);
                        rms_norm(row, readers.key_norm, row, epsilon);
                    }
                    apply_rope(query, head_dim, rotary_dim, scratch.rope_cosine, scratch.rope_sine);
                    apply_rope(scratch.key, head_dim, rotary_dim,
                               scratch.rope_cosine, scratch.rope_sine);
                    auto& cache = *attention_caches[layer];
                    cache.append(position, scratch.key.data(), scratch.key.size(),
                                 scratch.value.data(), scratch.value.size());
                    cache.attend(query.data(), query.size(), scratch.attended.data(),
                                 attention_heads * value_dim);
                    const auto gated_width = attention_heads * value_dim;
                    for (std::size_t i = 0; i < gated_width; ++i) {
                        scratch.attended[i] *= sigmoid(gate[i]);
                    }
                    tensor_linear(*readers.attention_output,
                        std::span<const float>(scratch.attended.data(), gated_width),
                        scratch.mixer_output, 0, &scratch.linear);
                } else {
                    const auto qk_width = checked_product(linear_qk_heads, linear_qk_dim,
                                                           "linear query/key");
                    const auto value_width = checked_product(linear_v_heads, linear_v_dim,
                                                              "linear value");
                    const auto qkv_width = checked_product(2, qk_width, "linear QKV") + value_width;
                    tensor_linear(*readers.linear_qkv, normalized,
                        std::span<float>(scratch.projected.data(), qkv_width), 0, &scratch.linear);
                    tensor_linear(*readers.linear_gate, normalized,
                        std::span<float>(scratch.gate.data(), value_width), 0, &scratch.linear);
                    auto& state = linear_states[layer];
                    causal_depthwise_conv1d_step(scratch.projected.data(),
                        readers.linear_convolution.data(), qkv_width, conv_kernel,
                        state.convolution.data(), scratch.projected.data());
                    auto* query = scratch.projected.data();
                    auto* key = query + qk_width;
                    auto* value = key + qk_width;
                    tensor_linear(*readers.linear_alpha, normalized,
                        std::span<float>(scratch.ffn_gate.data(), linear_v_heads), 0, &scratch.linear);
                    tensor_linear(*readers.linear_beta, normalized,
                        std::span<float>(scratch.ffn_up.data(), linear_v_heads), 0, &scratch.linear);
                    for (std::size_t head = 0; head < linear_v_heads; ++head) {
                        // GGUF stores ssm_a as -exp(A_log) (the converter folds
                        // this transformation into the tensor). Apply the
                        // timestep softplus directly, as the reference graph does.
                        const float decay = readers.linear_a[head] *
                            softplus(scratch.ffn_gate[head] + readers.linear_dt_bias[head]);
                        scratch.ffn_gate[head] = decay;
                        scratch.ffn_up[head] = sigmoid(scratch.ffn_up[head]);
                    }
                    gated_delta_recurrent_step_grouped(query, key, value,
                        scratch.ffn_gate.data(), scratch.ffn_up.data(), linear_qk_heads,
                        linear_v_heads, linear_qk_dim, linear_v_dim,
                        state.recurrent.data(), scratch.attended.data(), state.scratch);
                    rms_norm_gated(
                        std::span<const float>(scratch.attended.data(), value_width),
                        readers.linear_norm,
                        std::span<const float>(scratch.gate.data(), value_width),
                        std::span<float>(scratch.attended.data(), value_width),
                        linear_v_heads, epsilon);
                    tensor_linear(*readers.linear_output,
                        std::span<const float>(scratch.attended.data(), value_width),
                        scratch.mixer_output, 0, &scratch.linear);
                }

                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += scratch.mixer_output[i];
                rms_norm(hidden, readers.post_attention_norm, normalized, epsilon);
                if (!config.mixture_of_experts) {
                    tensor_linear(*readers.feed_forward_gate, normalized, scratch.ffn_gate,
                                  0, &scratch.linear);
                    tensor_linear(*readers.feed_forward_up, normalized, scratch.ffn_up,
                                  0, &scratch.linear);
                    for (std::size_t i = 0; i < scratch.ffn_activated.size(); ++i) {
                        scratch.ffn_activated[i] = silu(scratch.ffn_gate[i]) * scratch.ffn_up[i];
                    }
                    tensor_linear(*readers.feed_forward_down, scratch.ffn_activated,
                                  scratch.mixer_output, 0, &scratch.linear);
                } else {
                    const auto expert_width = as_size(config.expert_feed_forward_length,
                                                      "expert_feed_forward_length");
                    const auto shared_width = as_size(config.shared_expert_feed_forward_length,
                                                      "shared_expert_feed_forward_length");
                    const auto top_k = as_size(config.experts_per_token, "experts_per_token");

                    // Strata-inspired routing lookahead warms only OS file pages.
                    // The exact next-layer router result below remains authoritative.
                    if (layer + 1 < layers.size()) {
                        tensor_linear(*layers[layer + 1].router, normalized,
                                      scratch.router_logits, 0, &scratch.linear);
                        const auto predicted = select_experts(scratch.router_logits, top_k, false);
                        auto& future = routed_expert_stores[layer + 1];
                        for (const auto& route : predicted) {
                            future.gate->prefetch(route.id);
                            future.up->prefetch(route.id);
                            future.down->prefetch(route.id);
                        }
                    }

                    tensor_linear(*readers.router, normalized, scratch.router_logits,
                                  0, &scratch.linear);
                    const auto routes = select_experts(scratch.router_logits, top_k,
                                                        config.expert_weights_norm);
                    std::fill(scratch.moe_result.begin(), scratch.moe_result.end(), 0.0F);
                    auto& expert_stores = routed_expert_stores[layer];
                    const auto load_expert = [this, layer](std::size_t expert_id) {
                        return PendingExpertWeights{
                            expert_loader->submit([this, layer, expert_id] {
                                return routed_expert_stores[layer].gate->get(expert_id);
                            }),
                            expert_loader->submit([this, layer, expert_id] {
                                return routed_expert_stores[layer].up->get(expert_id);
                            }),
                            expert_loader->submit([this, layer, expert_id] {
                                return routed_expert_stores[layer].down->get(expert_id);
                            })};
                    };
                    auto pending = load_expert(routes.front().id);
                    const float expert_scale = static_cast<float>(config.expert_weights_scale);
                    for (std::size_t route_index = 0; route_index < routes.size(); ++route_index) {
                        const auto& route = routes[route_index];
                        auto gate_bytes = pending.gate.get();
                        auto up_bytes = pending.up.get();
                        auto down_bytes = pending.down.get();
                        if (route_index + 1 < routes.size()) {
                            pending = load_expert(routes[route_index + 1].id);
                        }
                        TensorReader gate_reader(
                            expert_view(index.layers()[layer].expert_gate,
                                        expert_stores.slices[0][route.id]), gate_bytes);
                        TensorReader up_reader(
                            expert_view(index.layers()[layer].expert_up,
                                        expert_stores.slices[1][route.id]), up_bytes);
                        TensorReader down_reader(
                            expert_view(index.layers()[layer].expert_down,
                                        expert_stores.slices[2][route.id]), down_bytes);
                        auto gate = std::span<float>(scratch.ffn_gate).first(expert_width);
                        auto up = std::span<float>(scratch.ffn_up).first(expert_width);
                        auto activated = std::span<float>(scratch.ffn_activated).first(expert_width);
                        tensor_linear(gate_reader, normalized, gate, 0, &scratch.linear);
                        tensor_linear(up_reader, normalized, up, 0, &scratch.linear);
                        for (std::size_t i = 0; i < expert_width; ++i) {
                            activated[i] = silu(gate[i]) * up[i];
                        }
                        tensor_linear(down_reader, activated, scratch.expert_output,
                                      0, &scratch.linear);
                        const float route_scale = route.weight * expert_scale;
                        for (std::size_t i = 0; i < hidden_size; ++i) {
                            scratch.moe_result[i] += route_scale * scratch.expert_output[i];
                        }
                    }

                    float shared_gate_value = 0.0F;
                    tensor_linear(*readers.shared_expert_router, normalized,
                                  std::span<float>(&shared_gate_value, 1), 0, &scratch.linear);
                    tensor_linear(*readers.shared_expert_gate, normalized,
                                  std::span<float>(scratch.ffn_gate).first(shared_width),
                                  0, &scratch.linear);
                    tensor_linear(*readers.shared_expert_up, normalized,
                                  std::span<float>(scratch.ffn_up).first(shared_width),
                                  0, &scratch.linear);
                    for (std::size_t i = 0; i < shared_width; ++i) {
                        scratch.ffn_activated[i] = silu(scratch.ffn_gate[i]) * scratch.ffn_up[i];
                    }
                    tensor_linear(*readers.shared_expert_down,
                                  std::span<const float>(scratch.ffn_activated).first(shared_width),
                                  scratch.expert_output, 0, &scratch.linear);
                    const float shared_scale = sigmoid(shared_gate_value);
                    for (std::size_t i = 0; i < hidden_size; ++i) {
                        scratch.mixer_output[i] = scratch.moe_result[i] +
                            shared_scale * scratch.expert_output[i];
                    }
                }
                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += scratch.mixer_output[i];
            }

            ++next_position;
            if (!calculate_logits) return;
            if (!logits) throw std::invalid_argument("Qwen3.5 logits output is required");
            rms_norm(scratch.hidden, output_norm, scratch.final_hidden, epsilon);
            logits->resize(as_size(index.vocabulary_size(), "vocabulary_size"));
            tensor_linear(output_reader, scratch.final_hidden, *logits, 0, &scratch.linear);
        } catch (...) {
            reset();
            throw;
        }
    }

    void reset() {
        for (auto& cache : attention_caches) if (cache) cache->clear();
        for (auto& state : linear_states) {
            std::fill(state.convolution.begin(), state.convolution.end(), 0.0F);
            std::fill(state.recurrent.begin(), state.recurrent.end(), 0.0F);
        }
        next_position = 0;
    }

    std::size_t kv_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto& cache : attention_caches) if (cache) bytes += cache->storage_bytes();
        return bytes;
    }

    std::size_t recurrent_bytes() const noexcept {
        std::size_t bytes = 0;
        for (const auto& state : linear_states) {
            bytes += (state.convolution.size() + state.recurrent.size()) * sizeof(float);
        }
        return bytes;
    }

    Qwen35ExpertCacheStats expert_cache_stats() const {
        Qwen35ExpertCacheStats stats;
        const auto add = [&stats](const storage::ExpertStore& store) {
            stats.capacity_bytes += store.cache_capacity_bytes();
            stats.resident_bytes += store.cache_bytes();
            stats.bytes_read += store.bytes_read_from_disk();
            stats.read_operations += store.read_operations();
            stats.read_time_ns += store.read_time_ns();
            stats.hits += store.cache_hits();
            stats.misses += store.cache_misses();
        };
        for (const auto& stores : routed_expert_stores) {
            add(*stores.gate);
            add(*stores.up);
            add(*stores.down);
        }
        return stats;
    }
};

Qwen35Runner::Qwen35Runner(std::filesystem::path model_path,
                           std::size_t context_capacity,
                           KvCachePrecision kv_precision)
    : Qwen35Runner(std::move(model_path), context_capacity, kv_precision, 3) {}

Qwen35Runner::Qwen35Runner(std::filesystem::path model_path,
                           std::size_t context_capacity,
                           KvCachePrecision kv_precision,
                           std::size_t expert_io_threads)
    : impl_(std::make_unique<Impl>(std::move(model_path), context_capacity, kv_precision,
                                   expert_io_threads)) {}
Qwen35Runner::~Qwen35Runner() = default;
Qwen35Runner::Qwen35Runner(Qwen35Runner&&) noexcept = default;
Qwen35Runner& Qwen35Runner::operator=(Qwen35Runner&&) noexcept = default;

std::vector<float> Qwen35Runner::forward_token(std::uint32_t token_id, std::uint64_t position) {
    if (!impl_) throw std::logic_error("Qwen3.5 runner has been moved from");
    std::vector<float> logits;
    impl_->forward_into(token_id, position, true, &logits);
    return logits;
}

std::vector<std::uint32_t> Qwen35Runner::generate_tokens(
    std::span<const std::uint32_t> prompt, std::size_t max_new_tokens,
    Qwen35GenerationStats* stats, const Qwen35ProgressCallback& progress) {
    if (!impl_) throw std::logic_error("Qwen3.5 runner has been moved from");
    if (stats) *stats = {};
    impl_->reset();
    if (max_new_tokens == 0) return {};
    if (prompt.empty()) throw std::invalid_argument("Qwen3.5 generation prompt is empty");
    if (prompt.size() > impl_->model_context ||
        max_new_tokens > impl_->model_context - prompt.size()) {
        throw std::invalid_argument("Qwen3.5 prompt and requested generation exceed model context");
    }
    if (stats) stats->prompt_tokens = prompt.size();

    std::vector<float> logits;
    const auto prefill_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        const auto token_start = std::chrono::steady_clock::now();
        const bool last = i + 1 == prompt.size();
        impl_->forward_into(prompt[i], i, last, last ? &logits : nullptr);
        const auto token_end = std::chrono::steady_clock::now();
        if (progress) {
            progress({Qwen35ProgressEvent::Phase::PrefillToken, i + 1, prompt[i],
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    token_end - token_start).count())});
        }
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->prefill_time_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(prefill_end - prefill_start).count());
    }

    std::vector<std::uint32_t> generated;
    generated.reserve(max_new_tokens);
    auto decode_start = prefill_end;
    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        const auto best = std::max_element(logits.begin(), logits.end());
        if (best == logits.end()) throw std::runtime_error("Qwen3.5 produced no vocabulary logits");
        const auto token = static_cast<std::uint32_t>(best - logits.begin());
        generated.push_back(token);
        if (progress) {
            progress({Qwen35ProgressEvent::Phase::GeneratedToken, generated.size(), token, 0});
        }
        if (stats) {
            stats->generated_tokens = generated.size();
            if (i == 0) {
                stats->first_generated_token_id = token;
                stats->has_generated_token = true;
            }
        }
        if (token == impl_->tokenizer.eos_token_id() || i + 1 == max_new_tokens) break;
        const auto forward_start = std::chrono::steady_clock::now();
        impl_->forward_into(token, prompt.size() + i, true, &logits);
        const auto forward_end = std::chrono::steady_clock::now();
        if (stats) ++stats->decode_forward_tokens;
        if (progress) {
            progress({Qwen35ProgressEvent::Phase::DecodeForward, generated.size(), token,
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    forward_end - forward_start).count())});
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->decode_time_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end - decode_start).count());
    }
    return generated;
}

std::string Qwen35Runner::generate_text(std::string_view prompt,
                                       std::size_t max_new_tokens,
                                       Qwen35GenerationStats* stats,
                                       const Qwen35ProgressCallback& progress) {
    if (!impl_) throw std::logic_error("Qwen3.5 runner has been moved from");
    const std::string text(prompt);
    const auto prompt_tokens = impl_->tokenizer.encode(text);
    auto generated = generate_tokens(prompt_tokens, max_new_tokens, stats, progress);
    generated.erase(std::remove(generated.begin(), generated.end(), impl_->tokenizer.eos_token_id()),
                    generated.end());
    return impl_->tokenizer.decode(generated);
}

std::string Qwen35Runner::generate_chat(std::string_view user_message,
                                       std::size_t max_new_tokens,
                                       Qwen35GenerationStats* stats,
                                       const Qwen35ProgressCallback& progress) {
    std::string prompt = "<|im_start|>user\n";
    prompt.append(user_message);
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return generate_text(prompt, max_new_tokens, stats, progress);
}

void Qwen35Runner::reset() { if (impl_) impl_->reset(); }
void Qwen35Runner::set_linear_backend(std::shared_ptr<TensorLinearBackend> backend) {
    if (!impl_) throw std::logic_error("Qwen3.5 runner has been moved from");
    impl_->linear_backend = std::move(backend);
    impl_->scratch.linear.backend = impl_->linear_backend.get();
}
const Qwen35Config& Qwen35Runner::config() const noexcept {
    static const Qwen35Config empty{};
    return impl_ ? impl_->index.config() : empty;
}
std::size_t Qwen35Runner::kv_cache_storage_bytes() const noexcept {
    return impl_ ? impl_->kv_bytes() : 0;
}
std::size_t Qwen35Runner::recurrent_state_storage_bytes() const noexcept {
    return impl_ ? impl_->recurrent_bytes() : 0;
}
std::uint64_t Qwen35Runner::bytes_read_from_disk() const noexcept {
    return impl_ ? impl_->file->bytes_read() : 0;
}
Qwen35ExpertCacheStats Qwen35Runner::expert_cache_stats() const {
    return impl_ ? impl_->expert_cache_stats() : Qwen35ExpertCacheStats{};
}

}  // namespace pokitlms::model
