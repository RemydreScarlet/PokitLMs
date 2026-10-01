#pragma once

#include "pokitlms/model/qwen35_index.hpp"
#include "pokitlms/ops/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace pokitlms::model {

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
