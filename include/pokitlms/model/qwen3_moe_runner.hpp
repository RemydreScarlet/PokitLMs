#pragma once

#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/model/qwen3_moe_index.hpp"
#include "pokitlms/storage/expert_store.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pokitlms::model {

struct GenerationOptions {
    float temperature{};                 // 0 selects greedy decoding.
    std::size_t top_k{};                 // 0 keeps the full vocabulary.
    float top_p{1.0F};                   // Nucleus threshold in (0, 1].
    float repetition_penalty{1.0F};      // 1 disables the penalty.
    std::uint64_t seed{};
};

struct ExpertCacheStats {
    std::size_t capacity_bytes{};
    std::size_t resident_bytes{};
    std::uint64_t bytes_read{};
    std::uint64_t hits{};
    std::uint64_t misses{};
};

// Single-sequence Qwen3-MoE decoder. The model tensors remain file-backed;
// only KV state, small norm vectors, and the selected experts are resident.
class Qwen3MoeRunner {
public:
    explicit Qwen3MoeRunner(std::filesystem::path model_path,
                            std::size_t expert_cache_budget_bytes = 128U * 1024U * 1024U,
                            std::size_t context_capacity = 0);
    ~Qwen3MoeRunner();
    Qwen3MoeRunner(Qwen3MoeRunner&&) noexcept;
    Qwen3MoeRunner& operator=(Qwen3MoeRunner&&) noexcept;
    Qwen3MoeRunner(const Qwen3MoeRunner&) = delete;
    Qwen3MoeRunner& operator=(const Qwen3MoeRunner&) = delete;

    // Consumes one token at its absolute sequence position and returns logits.
    [[nodiscard]] std::vector<float> forward_token(std::uint32_t token_id,
                                                   std::uint64_t position);
    // Consumes a tokenized prompt, resets existing KV state, and returns generated IDs.
    // The EOS token is included when generated.
    [[nodiscard]] std::vector<std::uint32_t> generate_tokens(
        std::span<const std::uint32_t> prompt, std::size_t max_new_tokens,
        const GenerationOptions& options = {});
    // Tokenizes a raw prompt and decodes generated IDs. Add any desired model chat
    // template markers to prompt before calling; this method does not infer a template.
    [[nodiscard]] std::string generate_text(
        std::string_view prompt, std::size_t max_new_tokens,
        const GenerationOptions& options = {});
    // Formats a single user message with Qwen3's standard text chat markers,
    // generates an assistant reply, and removes any generated end markers.
    [[nodiscard]] std::string generate_chat(
        std::string_view user_message, std::size_t max_new_tokens,
        const GenerationOptions& options = {});
    void reset();
    [[nodiscard]] const TransformerConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read_from_disk() const noexcept;
    [[nodiscard]] ExpertCacheStats expert_cache_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::model
