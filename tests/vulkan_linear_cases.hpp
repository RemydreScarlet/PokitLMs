#pragma once

#include "pokitlms/gpu/vulkan_linear.hpp"

#include <bit>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

// Shared by the host Vulkan test and the opt-in Android diagnostics. Uses
// nonzero scales, mixed signed values, multiple blocks, partial output groups,
// and many alternating stream windows; compares to the existing CPU kernels.
inline std::string run_vulkan_linear_checks() {
    std::ostringstream report;
    std::uint32_t state = 0x5935a3bU;
    const auto random = [&state]() {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return state;
    };
    for (const auto [subgroup, vectorized_q4_k] :
         {std::pair{true, true}, std::pair{true, false}, std::pair{false, true}}) {
        pokitlms::gpu::VulkanLinearOptions options;
        options.tile_bytes = 4096;
        options.use_subgroups = subgroup;
        options.use_vectorized_q4_k = vectorized_q4_k;
        pokitlms::gpu::VulkanLinearBackend backend(options);
        report << "device=" << backend.device_name() << '\n';
        for (std::uint32_t type : {0U, 1U, 30U, 2U, 3U, 8U, 12U, 13U, 14U}) {
            const std::size_t columns = type == 0 || type == 1 || type == 30 ? 257 : 512;
            constexpr std::size_t rows = 37;
            const std::size_t block = type >= 12 && type != 30 ? 256 : 32;
            const std::size_t bytes = type == 2 ? 18 : type == 3 ? 20 : type == 8 ? 34 :
                type == 12 ? 144 : type == 13 ? 176 : 210;
            const std::size_t row_bytes = type == 0 ? columns * 4 :
                type == 1 || type == 30 ? columns * 2 : columns / block * bytes;
            auto payload = std::make_shared<std::vector<std::byte>>(rows * row_bytes);
            for (auto& b : *payload) b = static_cast<std::byte>(random() & 255U);
            auto put_half = [&payload](std::size_t offset, std::uint16_t value) {
                (*payload)[offset] = static_cast<std::byte>(value & 255U);
                (*payload)[offset + 1] = static_cast<std::byte>(value >> 8);
            };
            if (type == 0) {
                for (std::size_t i = 0; i < rows * columns; ++i) {
                    float value = static_cast<float>(static_cast<int>(random() % 2049) - 1024) / 1024.0F;
                    std::memcpy(payload->data() + i * 4, &value, sizeof(value));
                }
            } else if (type == 1 || type == 30) {
                for (std::size_t i = 0; i < rows * columns; ++i) {
                    // Finite normal values with both signs; no NaN payloads.
                    const auto bits = type == 1 ? 0x3400U + (random() & 0x3ffU) : 0x3e80U + (random() & 0x7fU);
                    put_half(i * 2, static_cast<std::uint16_t>(bits | (random() & 0x8000U)));
                }
            } else {
                for (std::size_t r = 0; r < rows; ++r) {
                    for (std::size_t b = 0; b < columns / block; ++b) {
                        const auto offset = r * row_bytes + b * bytes;
                        put_half(offset + (type == 14 ? 208 : 0), 0x2800U);
                        if (type == 3 || type == 12 || type == 13) put_half(offset + 2, 0x2400U);
                    }
                }
            }
            pokitlms::model::TensorInfo tensor;
            tensor.name = "parity";
            tensor.dimensions = {columns, rows};
            tensor.type = type;
            tensor.payload_size = payload->size();
            pokitlms::model::TensorReader reader(tensor, payload);
            std::vector<float> input(columns), cpu(rows), gpu(rows);
            for (auto& value : input) value = static_cast<float>(static_cast<int>(random() % 2049) - 1024) / 1024.0F;
            pokitlms::model::tensor_linear(reader, input, cpu);
            double max_error = 0;
            // Repeat with different input so stale buffers or fences cannot pass.
            for (int repeat = 0; repeat < 3; ++repeat) {
                if (!backend.try_linear(reader, input, gpu)) throw std::runtime_error("supported Vulkan type was rejected");
                for (std::size_t i = 0; i < rows; ++i) {
                    const double error = std::abs(static_cast<double>(cpu[i]) - gpu[i]);
                    max_error = std::max(max_error, error);
                    if (!std::isfinite(gpu[i]) || error > 0.001 + 0.00002 * std::abs(cpu[i])) {
                        throw std::runtime_error("Vulkan/CPU parity failed, type=" + std::to_string(type) +
                            " row=" + std::to_string(i) + " error=" + std::to_string(error));
                    }
                }
                for (auto& value : input) value *= -0.75F;
                pokitlms::model::tensor_linear(reader, input, cpu);
            }
            report << "type=" << type << " max_abs_error=" << max_error << '\n';
        }
        const auto stats = backend.stats();
        if (stats.dispatches <= stats.linear_calls || stats.linear_calls != 27) {
            throw std::runtime_error("Vulkan tests did not exercise stream window reuse");
        }
        const bool subgroup_size_supported = backend.device_name().find("subgroup=16") != std::string::npos ||
                                             backend.device_name().find("subgroup=32") != std::string::npos;
        const auto expected_vectorized = subgroup && vectorized_q4_k && subgroup_size_supported ? 6U : 0U;
        if (stats.vectorized_q4_k_calls != expected_vectorized) {
            throw std::runtime_error("Vulkan test did not exercise the requested Q4_K/Q5_K path");
        }
        report << "calls=" << stats.linear_calls << " dispatches=" << stats.dispatches
               << " vectorized_q4_k_calls=" << stats.vectorized_q4_k_calls
               << " allocated_bytes=" << stats.allocated_bytes << '\n';
    }
#ifdef POKITLMS_GPU_DIAGNOSTICS
    {
        // A small, model-free Q6_K workload makes shader changes measurable
        // on Android without loading a multi-gigabyte GGUF.
        constexpr std::size_t rows = 2048;
        constexpr std::size_t columns = 512;
        constexpr std::size_t row_bytes = 420;
        auto payload = std::make_shared<std::vector<std::byte>>(rows * row_bytes);
        for (auto& byte : *payload) byte = static_cast<std::byte>(random() & 255U);
        const auto put_half = [&payload](std::size_t offset, std::uint16_t value) {
            (*payload)[offset] = static_cast<std::byte>(value & 255U);
            (*payload)[offset + 1] = static_cast<std::byte>(value >> 8);
        };
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t block = 0; block < columns / 256; ++block)
                put_half(row * row_bytes + block * 210 + 208, 0x2800U);
        }
        pokitlms::model::TensorInfo tensor;
        tensor.name = "q6_microbench";
        tensor.dimensions = {columns, rows};
        tensor.type = 14;
        tensor.payload_size = payload->size();
        pokitlms::model::TensorReader reader(tensor, payload);
        std::vector<float> input(columns), cpu(rows), gpu(rows);
        for (auto& value : input)
            value = static_cast<float>(static_cast<int>(random() % 2049) - 1024) / 1024.0F;
        pokitlms::model::tensor_linear(reader, input, cpu);

        pokitlms::gpu::VulkanLinearOptions options;
        options.tile_bytes = 1024U * 1024U;
        options.use_subgroups = true;
        pokitlms::gpu::VulkanLinearBackend backend(options);
        for (int i = 0; i < 8; ++i) {
            if (!backend.try_linear(reader, input, gpu))
                throw std::runtime_error("Q6_K microbenchmark type was rejected");
        }
        const auto check_output = [&] {
            double max_error = 0;
            for (std::size_t row = 0; row < rows; ++row) {
                const double error = std::abs(static_cast<double>(cpu[row]) - gpu[row]);
                max_error = std::max(max_error, error);
                if (!std::isfinite(gpu[row]) || error > 0.001 + 0.00002 * std::abs(cpu[row]))
                    throw std::runtime_error("Q6_K microbenchmark CPU/GPU parity failed");
            }
            return max_error;
        };
        double max_error = check_output();
        constexpr std::uint64_t measured_calls = 32;
        constexpr std::uint64_t measured_batches = 6;
        report << "q6_microbench_rows=" << rows << " columns=" << columns
               << " warmup_calls=8 batches=" << measured_batches
               << " calls_per_batch=" << measured_calls << '\n';
        for (std::uint64_t batch = 0; batch < measured_batches; ++batch) {
            const auto before = backend.stats();
            for (std::uint64_t i = 0; i < measured_calls; ++i) {
                if (!backend.try_linear(reader, input, gpu))
                    throw std::runtime_error("Q6_K microbenchmark type was rejected");
            }
            const auto after = backend.stats();
            if (after.q6_k_dispatches - before.q6_k_dispatches < measured_calls)
                throw std::runtime_error("Q6_K microbenchmark did not dispatch each measured call");
            const auto gpu_time_ns = after.q6_k_gpu_time_ns - before.q6_k_gpu_time_ns;
            max_error = std::max(max_error, check_output());
            report << "q6_microbench_batch=" << batch
                   << " q6_gpu_ms=" << static_cast<double>(gpu_time_ns) / 1'000'000.0
                   << " q6_gpu_us_per_call=" << static_cast<double>(gpu_time_ns) /
                        static_cast<double>(measured_calls) / 1000.0
                   << " max_abs_error=" << max_error << '\n';
        }
    }
#endif
    return "PASS\n" + report.str();
}
