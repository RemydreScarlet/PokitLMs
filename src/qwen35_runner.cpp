#include "pokitlms/model/qwen35_runner.hpp"

#include "pokitlms/model/qwen_bpe_tokenizer.hpp"
#include "pokitlms/model/tensor_linear.hpp"
#include "pokitlms/model/tensor_reader.hpp"
#include "pokitlms/ops/causal_conv1d.hpp"
#include "pokitlms/ops/gated_delta_net.hpp"
#include "pokitlms/ops/quantized_linear.hpp"
#include "pokitlms/ops/rms_norm.hpp"
#include "simd_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

void prepare_rope(std::uint64_t position, std::size_t rotary_dimension, float theta,
                  std::span<float> cosine, std::span<float> sine) {
    if (rotary_dimension == 0 || rotary_dimension % 2 != 0 ||
        cosine.size() != rotary_dimension / 2 || sine.size() != cosine.size() ||
        !std::isfinite(theta) || theta <= 0.0F) {
        throw std::invalid_argument("invalid Qwen3.5 RoPE parameters");
    }
    const auto half = rotary_dimension / 2;
    for (std::size_t pair = 0; pair < half; ++pair) {
        const float exponent = -static_cast<float>(pair) / static_cast<float>(half);
        const float angle = static_cast<float>(position) * std::pow(theta, exponent);
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

struct LayerReaders {
    explicit LayerReaders(const std::shared_ptr<storage::ModelFile>& file,
                          const Qwen35LayerTensors& tensors)
        : full_attention(tensors.full_attention),
          feed_forward_gate(file, tensors.feed_forward_gate),
          feed_forward_up(file, tensors.feed_forward_up),
          feed_forward_down(file, tensors.feed_forward_down),
          attention_norm(load_vector(file, tensors.attention_norm)),
          post_attention_norm(load_vector(file, tensors.post_attention_norm)) {
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
    TensorReader feed_forward_gate;
    TensorReader feed_forward_up;
    TensorReader feed_forward_down;
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
    std::vector<float> final_hidden;
    std::vector<float> rope_cosine;
    std::vector<float> rope_sine;
    std::vector<std::byte> embedding;
    TensorLinearScratch linear;
};

}  // namespace

class Qwen35Runner::Impl {
public:
    Impl(std::filesystem::path path, std::size_t requested_context,
         KvCachePrecision kv_precision)
        : model_path(std::move(path)), gguf(model_path),
          file(std::make_shared<storage::ModelFile>(model_path)), index(gguf), tokenizer(gguf),
          embedding_reader(file, index.token_embedding()), output_reader(file, index.output()),
          output_norm(load_vector(file, index.output_norm())) {
        const auto& config = index.config();
        model_context = as_size(config.context_length, "context_length");
        context_capacity = requested_context == 0 ? std::min<std::size_t>(model_context, 512)
                                                   : requested_context;
        if (context_capacity == 0 || context_capacity > model_context) {
            throw std::invalid_argument("requested KV window exceeds Qwen3.5 model context");
        }

        const auto hidden = as_size(config.embedding_length, "embedding_length");
        const auto ffn = as_size(config.feed_forward_length, "feed_forward_length");
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
        scratch.final_hidden.resize(hidden);
        scratch.embedding.resize(embedding_reader.row_bytes());
        scratch.rope_cosine.resize(rotary_dim / 2);
        scratch.rope_sine.resize(rotary_dim / 2);

        layers.reserve(index.layers().size());
        linear_states.resize(index.layers().size());
        attention_caches.resize(index.layers().size());
        for (std::size_t layer = 0; layer < index.layers().size(); ++layer) {
            const auto& tensors = index.layers()[layer];
            layers.emplace_back(file, tensors);
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
    std::vector<LinearState> linear_states;
    std::vector<std::unique_ptr<KvCache>> attention_caches;
    std::size_t model_context{};
    std::size_t context_capacity{};
    std::uint64_t next_position{};
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
            prepare_rope(position, rotary_dim, theta, scratch.rope_cosine, scratch.rope_sine);

            for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                auto& readers = layers[layer];
                auto& hidden = scratch.hidden;
                auto& normalized = scratch.normalized;
                rms_norm_zero_centered(hidden, readers.attention_norm, normalized, epsilon);
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
                        rms_norm_zero_centered(row, readers.query_norm, row, epsilon);
                    }
                    for (std::size_t head = 0; head < kv_heads; ++head) {
                        auto row = std::span<float>(scratch.key).subspan(head * head_dim, head_dim);
                        rms_norm_zero_centered(row, readers.key_norm, row, epsilon);
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
                        const float decay = -std::exp(readers.linear_a[head]) *
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
                rms_norm_zero_centered(hidden, readers.post_attention_norm, normalized, epsilon);
                tensor_linear(readers.feed_forward_gate, normalized, scratch.ffn_gate,
                              0, &scratch.linear);
                tensor_linear(readers.feed_forward_up, normalized, scratch.ffn_up,
                              0, &scratch.linear);
                for (std::size_t i = 0; i < scratch.ffn_activated.size(); ++i) {
                    scratch.ffn_activated[i] = silu(scratch.ffn_gate[i]) * scratch.ffn_up[i];
                }
                tensor_linear(readers.feed_forward_down, scratch.ffn_activated,
                              scratch.mixer_output, 0, &scratch.linear);
                for (std::size_t i = 0; i < hidden_size; ++i) hidden[i] += scratch.mixer_output[i];
            }

            ++next_position;
            if (!calculate_logits) return;
            if (!logits) throw std::invalid_argument("Qwen3.5 logits output is required");
            rms_norm_zero_centered(scratch.hidden, output_norm, scratch.final_hidden, epsilon);
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
};

Qwen35Runner::Qwen35Runner(std::filesystem::path model_path,
                           std::size_t context_capacity,
                           KvCachePrecision kv_precision)
    : impl_(std::make_unique<Impl>(std::move(model_path), context_capacity, kv_precision)) {}
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
    Qwen35GenerationStats* stats) {
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
        const bool last = i + 1 == prompt.size();
        impl_->forward_into(prompt[i], i, last, last ? &logits : nullptr);
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
        if (stats) stats->generated_tokens = generated.size();
        if (token == impl_->tokenizer.eos_token_id() || i + 1 == max_new_tokens) break;
        impl_->forward_into(token, prompt.size() + i, true, &logits);
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
                                       Qwen35GenerationStats* stats) {
    if (!impl_) throw std::logic_error("Qwen3.5 runner has been moved from");
    const std::string text(prompt);
    const auto prompt_tokens = impl_->tokenizer.encode(text);
    auto generated = generate_tokens(prompt_tokens, max_new_tokens, stats);
    generated.erase(std::remove(generated.begin(), generated.end(), impl_->tokenizer.eos_token_id()),
                    generated.end());
    return impl_->tokenizer.decode(generated);
}

std::string Qwen35Runner::generate_chat(std::string_view user_message,
                                       std::size_t max_new_tokens,
                                       Qwen35GenerationStats* stats) {
    std::string prompt = "<|im_start|>user\n";
    prompt.append(user_message);
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return generate_text(prompt, max_new_tokens, stats);
}

void Qwen35Runner::reset() { if (impl_) impl_->reset(); }
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

}  // namespace pokitlms::model
