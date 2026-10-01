#pragma once

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

// Reads routed expert weights from a model file while keeping RAM use bounded.
// The caller owns the model-specific index that maps expert IDs to slices.
class ExpertStore {
public:
    ExpertStore(std::filesystem::path path, std::vector<ExpertSlice> experts,
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
    void clear_cache();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::storage
