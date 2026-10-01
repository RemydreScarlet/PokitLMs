#pragma once

#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/storage/model_file.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace pokitlms::storage {

struct ExpertSlice {
    std::uint64_t offset{};
    std::size_t size{};
};

// Splits a GGUF expert tensor into equal, contiguous slices along its last
// (expert) dimension. Throws unless payload size and expert axis are known.
[[nodiscard]] std::vector<ExpertSlice> split_expert_tensor(
    const model::TensorInfo& tensor, std::size_t expert_count);

// Reads routed expert weights from a model file while keeping RAM use bounded.
// The caller owns the model-specific index that maps expert IDs to slices.
class ExpertStore {
public:
    ExpertStore(std::filesystem::path path, std::vector<ExpertSlice> experts,
                std::size_t cache_capacity_bytes);
    ExpertStore(std::shared_ptr<ModelFile> file, std::vector<ExpertSlice> experts,
                std::size_t cache_capacity_bytes);
    ~ExpertStore();

    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;
    ExpertStore(ExpertStore&&) noexcept;
    ExpertStore& operator=(ExpertStore&&) noexcept;

    [[nodiscard]] std::shared_ptr<const std::vector<std::byte>> get(std::size_t expert_id);
    [[nodiscard]] std::size_t cache_capacity_bytes() const;
    [[nodiscard]] std::size_t cache_bytes() const;
    [[nodiscard]] std::uint64_t bytes_read_from_disk() const;
    [[nodiscard]] std::uint64_t cache_hits() const;
    [[nodiscard]] std::uint64_t cache_misses() const;
    void clear_cache();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::storage
