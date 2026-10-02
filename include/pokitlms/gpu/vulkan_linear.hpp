#pragma once

#include "pokitlms/model/tensor_linear.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace pokitlms::gpu {

struct VulkanLinearOptions {
    // Two mapped weight windows are allocated. The complete model is never
    // uploaded: positioned reads into the next window overlap GPU execution.
    std::size_t tile_bytes = 16U * 1024U * 1024U;
    std::uint32_t device_index = std::numeric_limits<std::uint32_t>::max();
    bool use_subgroups = true;
    // Diagnostic comparison for discrete GPUs: cached system memory versus
    // mapped device memory. Integrated GPUs normally have cached local memory.
    bool prefer_host_cached_weights = false;
};

struct VulkanLinearStats {
    std::uint64_t linear_calls{};
    std::uint64_t dispatches{};
    std::uint64_t weight_bytes{};
    std::uint64_t read_time_ns{};
    std::uint64_t wait_time_ns{};
    std::uint64_t gpu_time_ns{};
    std::uint64_t pipeline_time_ns{};
    std::uint64_t cache_time_ns{};
    std::uint64_t allocated_bytes{};
    std::uint32_t weight_memory_flags{};
};

class VulkanLinearBackend final : public model::TensorLinearBackend {
public:
    explicit VulkanLinearBackend(const VulkanLinearOptions& options = {});
    ~VulkanLinearBackend() override;
    VulkanLinearBackend(const VulkanLinearBackend&) = delete;
    VulkanLinearBackend& operator=(const VulkanLinearBackend&) = delete;

    bool try_linear(const model::TensorReader& weights, std::span<const float> input,
                    std::span<float> output) override;
    [[nodiscard]] VulkanLinearStats stats() const;
    [[nodiscard]] std::string device_name() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pokitlms::gpu
