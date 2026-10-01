#pragma once

#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/model/qwen3_moe_index.hpp"
#include "pokitlms/storage/expert_store.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace pokitlms::model {

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
    void reset();
    [[nodiscard]] const TransformerConfig& config() const noexcept;
    [[nodiscard]] std::uint64_t bytes_read_from_disk() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::model
