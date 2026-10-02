#pragma once

#include "pokitlms/model/qwen35_index.hpp"
#include "pokitlms/ops/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pokitlms::model {

class TensorLinearBackend;

struct Qwen35GenerationStats {
    std::uint64_t prefill_time_ns{};
    std::uint64_t decode_time_ns{};
    std::size_t prompt_tokens{};
    std::size_t generated_tokens{};
    std::size_t decode_forward_tokens{};
    std::uint32_t first_generated_token_id{};
    bool has_generated_token{};
};

struct Qwen35ProgressEvent {
    // Forward phases report compute duration. GeneratedToken reports the
    // selected token immediately, with elapsed_ns == 0.
    enum class Phase {
        PrefillToken,
        GeneratedToken,
        DecodeForward,
    };

    Phase phase{};
    std::size_t index{};
    std::uint32_t token_id{};
    std::uint64_t elapsed_ns{};
};

using Qwen35ProgressCallback = std::function<void(const Qwen35ProgressEvent&)>;

struct Qwen35ExpertCacheStats {
    std::size_t capacity_bytes{};
    std::size_t resident_bytes{};
    std::uint64_t bytes_read{};
    std::uint64_t read_operations{};
    std::uint64_t read_time_ns{};
    std::uint64_t hits{};
    std::uint64_t misses{};
};

// Single-sequence Qwen3.5 text decoder. Matrix weights are read from GGUF or
// cached in Vulkan device-local memory when the GPU budget permits; recurrent,
// convolution, and bounded full-attention state stay resident.
class Qwen35Runner {
public:
    explicit Qwen35Runner(std::filesystem::path model_path,
                          std::size_t context_capacity = 0,
                          KvCachePrecision kv_precision = KvCachePrecision::Float16);
    Qwen35Runner(std::filesystem::path model_path, std::size_t context_capacity,
                 KvCachePrecision kv_precision, std::size_t expert_io_threads);
    ~Qwen35Runner();
    Qwen35Runner(Qwen35Runner&&) noexcept;
    Qwen35Runner& operator=(Qwen35Runner&&) noexcept;
    Qwen35Runner(const Qwen35Runner&) = delete;
    Qwen35Runner& operator=(const Qwen35Runner&) = delete;

    [[nodiscard]] std::vector<float> forward_token(std::uint32_t token_id,
                                                   std::uint64_t position);
    // Greedy autoregressive generation from token IDs. The runner is reset
    // before the prompt is consumed and retains the resulting decode state.
    [[nodiscard]] std::vector<std::uint32_t> generate_tokens(
        std::span<const std::uint32_t> prompt, std::size_t max_new_tokens,
        Qwen35GenerationStats* stats = nullptr,
        const Qwen35ProgressCallback& progress = {});
    [[nodiscard]] std::string generate_text(std::string_view prompt,
                                            std::size_t max_new_tokens,
                                            Qwen35GenerationStats* stats = nullptr,
                                            const Qwen35ProgressCallback& progress = {});
    [[nodiscard]] std::string generate_chat(std::string_view user_message,
                                            std::size_t max_new_tokens,
                                            Qwen35GenerationStats* stats = nullptr,
                                            const Qwen35ProgressCallback& progress = {});
    void reset();
    // The runner owns the backend; unsupported projections retain CPU math.
    void set_linear_backend(std::shared_ptr<TensorLinearBackend> backend);
    [[nodiscard]] const Qwen35Config& config() const noexcept;
    [[nodiscard]] std::size_t kv_cache_storage_bytes() const noexcept;
    [[nodiscard]] std::size_t recurrent_state_storage_bytes() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read_from_disk() const noexcept;
    [[nodiscard]] Qwen35ExpertCacheStats expert_cache_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::model
