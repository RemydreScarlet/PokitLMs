#include "pokitlms/gpu/vulkan_linear.hpp"
#include "pokitlms/model/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::uint64_t physical_read_bytes() {
    std::ifstream file("/proc/self/io");
    std::string key;
    std::uint64_t value{};
    while (file >> key >> value) if (key == "read_bytes:") return value;
    return 0;
}
double compare(std::span<const float> cpu, std::span<const float> gpu) {
    double max_error = 0;
    for (std::size_t row = 0; row < cpu.size(); ++row) {
        const auto error = std::abs(static_cast<double>(gpu[row]) - cpu[row]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(gpu[row]) || error > 0.0001 + 0.00005 * std::abs(cpu[row]))
            throw std::runtime_error("matrix parity failed at row " + std::to_string(row) +
                                     ", error=" + std::to_string(error));
    }
    return max_error;
}
std::vector<float> input_for(std::size_t columns) {
    std::vector<float> input(columns);
    for (std::size_t i = 0; i < columns; ++i)
        input[i] = std::sin(static_cast<float>(i + 1) * 0.37F) * 0.7F;
    return input;
}
pokitlms::model::TensorInfo view(const pokitlms::model::TensorInfo& tensor,
                               std::uint64_t first, std::size_t rows,
                               std::size_t row_bytes) {
    auto result = tensor;
    result.dimensions = {tensor.dimensions.front(), rows};
    result.file_offset += first * row_bytes;
    result.payload_size = rows * row_bytes;
    return result;
}
}  // namespace

// Reads bounded slices and one repeatedly warmed matrix window by default.
// Whole-model residency is opt-in for diagnosing its exact matrix path.
int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " MODEL.gguf [timing_tensor=blk.0.ffn_gate.weight]"
                  << " [weights=auto|local|cached] [model_cache=off|auto|full] [gpu_device=auto|N]\n";
        return 2;
    }
    try {
        pokitlms::model::GgufReader gguf(argv[1]);
        auto file = std::make_shared<pokitlms::storage::ModelFile>(argv[1]);
        pokitlms::gpu::VulkanLinearOptions options;
        std::string_view memory = argc > 3 ? argv[3] : "auto";
        if (memory == "device") memory = "local";
        if (memory != "auto" && memory != "local" && memory != "cached")
            throw std::invalid_argument("weights must be auto, local, or cached");
        if (memory == "local") {
            options.weight_memory_mode = pokitlms::gpu::VulkanWeightMemoryMode::Local;
        } else if (memory == "cached") {
            options.weight_memory_mode = pokitlms::gpu::VulkanWeightMemoryMode::Cached;
        }
        std::string_view model_cache = argc > 4 ? argv[4] : "off";
        if (model_cache != "off" && model_cache != "auto" && model_cache != "full")
            throw std::invalid_argument("model_cache must be off, auto, or full");
        if (model_cache == "off")
            options.model_cache_mode = pokitlms::gpu::VulkanModelCacheMode::Disabled;
        std::string_view device = argc > 5 ? argv[5] : "auto";
        if (device != "auto") {
            std::uint32_t index{};
            const auto [end, error] = std::from_chars(device.data(), device.data() + device.size(), index);
            if (error != std::errc{} || end != device.data() + device.size())
                throw std::invalid_argument("gpu_device must be auto or a zero-based index");
            options.device_index = index;
        }
        pokitlms::gpu::VulkanLinearBackend backend(options);
        std::cout << "device=" << backend.device_name()
                  << " weight_memory_mode=" << memory
                  << " model_cache_mode=" << model_cache
                  << " weight_memory_flags=" << backend.stats().weight_memory_flags << '\n';
        std::map<std::uint32_t, double> errors;
        std::size_t matrices = 0, samples = 0;
        for (const auto& tensor : gguf.tensors()) {
            if (tensor.dimensions.size() < 2 || !tensor.payload_size ||
                tensor.dimensions.front() > 65536) continue;
            pokitlms::model::TensorReader original(file, tensor);
            const auto rows = std::min<std::size_t>(8, original.row_count());
            if (!rows) continue;
            auto input = input_for(tensor.dimensions.front());
            std::vector<float> cpu(rows), gpu(rows);
            bool used_gpu = false;
            for (const auto first : {std::uint64_t(0), (original.row_count() - rows) / 2,
                                    original.row_count() - rows}) {
                auto payload = std::make_shared<const std::vector<std::byte>>(original.read_rows(first, rows));
                auto slice = view(tensor, 0, rows, original.row_bytes());
                slice.file_offset = 0;
                pokitlms::model::TensorReader reader(slice, payload);
                if (!backend.try_linear(reader, input, gpu)) break;
                pokitlms::model::tensor_linear(reader, input, cpu);
                errors[tensor.type] = std::max(errors[tensor.type], compare(cpu, gpu));
                ++samples;
                used_gpu = true;
            }
            if (used_gpu) ++matrices;
        }
        std::cout << "matrix_parity=PASS matrices=" << matrices << " samples=" << samples << '\n';
        for (const auto& [type, error] : errors)
            std::cout << "type=" << type << " max_abs_error=" << error << '\n';

        const auto* tensor = gguf.find_tensor(argc > 2 ? argv[2] : "blk.0.ffn_gate.weight");
        if (!tensor) throw std::runtime_error("timing tensor not found");
        pokitlms::model::TensorReader original(file, *tensor);
        const bool full_tensor = model_cache == "full";
        if (full_tensor && original.row_count() > 16384)
            throw std::runtime_error("full-tensor probe is limited to 16384 output rows");
        const auto rows = full_tensor ? static_cast<std::size_t>(original.row_count())
            : std::min<std::size_t>({2048, original.row_count(),
                                     8U * 1024U * 1024U / original.row_bytes()});
        if (!rows) throw std::runtime_error("timing row exceeds window limit");
        auto slice = view(*tensor, 0, rows, original.row_bytes());
        pokitlms::model::TensorReader reader(file, slice);
        std::vector<std::byte> window(rows * reader.row_bytes());
        auto input = input_for(tensor->dimensions.front());
        std::vector<float> cpu(rows), gpu(rows);
        reader.read_rows_into(0, rows, window); // Warm only this bounded region.
        const unsigned repeats = full_tensor ? 1U : 8U;
        auto disk_before = physical_read_bytes();
        auto start = Clock::now();
        for (unsigned i = 0; i < repeats; ++i) reader.read_rows_into(0, rows, window);
        const auto read_ms = elapsed_ms(start);
        std::cout << "cpu_warm_read_bytes=" << window.size() * repeats
                  << " cpu_warm_read_ms=" << read_ms
                  << " physical_read_bytes=" << physical_read_bytes() - disk_before << '\n';
        auto payload = std::make_shared<const std::vector<std::byte>>(window);
        pokitlms::model::TensorReader resident(slice, payload);
        pokitlms::model::tensor_linear(resident, input, cpu);
        if (!backend.try_linear(reader, input, gpu)) throw std::runtime_error("timing type unsupported");
        const auto timing_error = compare(cpu, gpu);
        const auto before = backend.stats();
        disk_before = physical_read_bytes();
        start = Clock::now();
        for (unsigned i = 0; i < repeats; ++i) {
            backend.try_linear(reader, input, gpu);
            compare(cpu, gpu);
        }
        const auto total_ms = elapsed_ms(start);
        const auto after = backend.stats();
        std::cout << "gpu_warm_read_bytes=" << after.weight_bytes - before.weight_bytes
                  << " gpu_warm_read_ms=" << (after.read_time_ns - before.read_time_ns) / 1e6
                  << " gpu_compute_ms=" << (after.gpu_time_ns - before.gpu_time_ns) / 1e6
                  << " gpu_wait_ms=" << (after.wait_time_ns - before.wait_time_ns) / 1e6
                  << " gpu_cache_ms=" << (after.cache_time_ns - before.cache_time_ns) / 1e6
                  << " model_cache_active=" << after.model_cache_active
                  << " model_cache_uploaded_bytes=" << after.model_cache_uploaded_bytes
                  << " model_cache_upload_ms=" << after.model_cache_upload_time_ns / 1e6
                  << " timing_matrix_max_abs_error=" << timing_error
                  << " full_tensor=" << full_tensor
                  << " total_ms=" << total_ms
                  << " physical_read_bytes=" << physical_read_bytes() - disk_before << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vulkan-probe: " << error.what() << '\n';
        return 1;
    }
}
