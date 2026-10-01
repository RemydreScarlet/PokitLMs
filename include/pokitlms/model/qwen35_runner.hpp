#pragma once

#include "pokitlms/model/qwen35_index.hpp"
#include "pokitlms/ops/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pokitlms::model {

struct Qwen35GenerationStats {
    std::uint64_t prefill_time_ns{};
    std::uint64_t decode_time_ns{};
    std::size_t prompt_tokens{};
    std::size_t generated_tokens{};
};

// Single-sequence Qwen3.5 text decoder. Matrix weights stay file-backed;
// recurrent, convolution, and bounded full-attention state stay resident.
class Qwen35Runner {
public:
    explicit Qwen35Runner(std::filesystem::path model_path,
                          std::size_t context_capacity = 0,
                          KvCachePrecision kv_precision = KvCachePrecision::Float16);
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
        Qwen35GenerationStats* stats = nullptr);
    [[nodiscard]] std::string generate_text(std::string_view prompt,
                                            std::size_t max_new_tokens,
                                            Qwen35GenerationStats* stats = nullptr);
    [[nodiscard]] std::string generate_chat(std::string_view user_message,
                                            std::size_t max_new_tokens,
                                            Qwen35GenerationStats* stats = nullptr);
    void reset();
    [[nodiscard]] const Qwen35Config& config() const noexcept;
    [[nodiscard]] std::size_t kv_cache_storage_bytes() const noexcept;
    [[nodiscard]] std::size_t recurrent_state_storage_bytes() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read_from_disk() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::model
