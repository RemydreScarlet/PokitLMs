#include "pokitlms/model/qwen35_runner.hpp"
#ifdef POKITLMS_USE_VULKAN
#include "pokitlms/gpu/vulkan_linear.hpp"
#endif

#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <fstream>
#include <optional>
#include <sys/resource.h>
#endif

namespace {

std::size_t parse_size(const char* text, const char* label) {
    const std::string_view input(text);
    std::uint64_t value{};
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size() ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::size_t>(value);
}

double milliseconds(std::uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1'000'000.0;
}

double tokens_per_second(std::size_t tokens, std::uint64_t nanoseconds) {
    if (nanoseconds == 0) return 0.0;
    return static_cast<double>(tokens) * 1'000'000'000.0 /
           static_cast<double>(nanoseconds);
}

#if defined(__linux__)
struct ProcessTelemetry {
    std::optional<std::uint64_t> rchar;
    std::optional<std::uint64_t> read_bytes;
    std::optional<std::uint64_t> rss_kb;
    std::optional<std::uint64_t> swap_kb;
    std::optional<std::uint64_t> mem_available_kb;
    std::optional<double> cpu_user_ms;
    std::optional<double> cpu_sys_ms;
};

template <typename Visitor>
void read_proc_values(const char* path, Visitor&& visitor) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string_view view(line);
        auto value_text = view.substr(colon + 1);
        const auto first = value_text.find_first_not_of(" \t");
        if (first == std::string_view::npos) continue;
        value_text.remove_prefix(first);
        std::uint64_t value{};
        const auto [end, error] = std::from_chars(
            value_text.data(), value_text.data() + value_text.size(), value);
        if (error == std::errc{} && end != value_text.data()) {
            visitor(view.substr(0, colon + 1), value);
        }
    }
}

ProcessTelemetry process_telemetry() {
    ProcessTelemetry result;
    read_proc_values("/proc/self/io", [&](std::string_view key, std::uint64_t value) {
        if (key == "rchar:") result.rchar = value;
        else if (key == "read_bytes:") result.read_bytes = value;
    });
    read_proc_values("/proc/self/status", [&](std::string_view key, std::uint64_t value) {
        if (key == "VmRSS:") result.rss_kb = value;
        else if (key == "VmSwap:") result.swap_kb = value;
    });
    read_proc_values("/proc/meminfo", [&](std::string_view key, std::uint64_t value) {
        if (key == "MemAvailable:") result.mem_available_kb = value;
    });
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) == 0) {
        result.cpu_user_ms = static_cast<double>(usage.ru_utime.tv_sec) * 1000.0 +
                             static_cast<double>(usage.ru_utime.tv_usec) / 1000.0;
        result.cpu_sys_ms = static_cast<double>(usage.ru_stime.tv_sec) * 1000.0 +
                            static_cast<double>(usage.ru_stime.tv_usec) / 1000.0;
    }
    return result;
}

void log_process_telemetry(const ProcessTelemetry& snapshot) {
    if (snapshot.rchar) std::cerr << " rchar=" << *snapshot.rchar;
    if (snapshot.read_bytes) std::cerr << " read_bytes=" << *snapshot.read_bytes;
    if (snapshot.rss_kb) std::cerr << " rss_kb=" << *snapshot.rss_kb;
    if (snapshot.swap_kb) std::cerr << " swap_kb=" << *snapshot.swap_kb;
    if (snapshot.mem_available_kb) std::cerr << " mem_available_kb=" << *snapshot.mem_available_kb;
    if (snapshot.cpu_user_ms) std::cerr << " cpu_user_ms=" << *snapshot.cpu_user_ms;
    if (snapshot.cpu_sys_ms) std::cerr << " cpu_sys_ms=" << *snapshot.cpu_sys_ms;
}
#endif

void usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " MODEL.gguf USER_MESSAGE [new_tokens=32] [kv_window=0] [kv_precision=fp16|q8] [io_threads=3]"
              << " [backend=cpu|vulkan] [gpu_tile_mib=16] [gpu_device=auto] [gpu_mode=subgroup|workgroup]"
              << " [gpu_weight_memory=auto|local|cached] [gpu_model_cache=auto|off]"
              << " [gpu_vectorized_q4=true|false]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 14) {
        usage(argv[0]);
        return 2;
    }
    try {
        const auto max_new_tokens = argc > 3 ? parse_size(argv[3], "new token count") : 32;
        const auto kv_window = argc > 4 ? parse_size(argv[4], "KV window") : 0;
        auto kv_precision = pokitlms::KvCachePrecision::Float16;
        if (argc > 5) {
            const std::string_view precision(argv[5]);
            if (precision == "q8") kv_precision = pokitlms::KvCachePrecision::Q8_0;
            else if (precision != "fp16") throw std::invalid_argument("KV precision must be fp16 or q8");
        }
        const auto io_threads = argc > 6 ? parse_size(argv[6], "expert I/O thread count") : 3;
        const std::string_view backend = argc > 7 ? argv[7] : "cpu";
        if (backend != "cpu" && backend != "vulkan")
            throw std::invalid_argument("backend must be cpu or vulkan");
        const auto tile_mib = argc > 8 ? parse_size(argv[8], "GPU tile MiB") : 16;
        if (tile_mib < 1 || tile_mib > 128)
            throw std::invalid_argument("GPU tile MiB must be between 1 and 128");
        const std::string_view device = argc > 9 ? argv[9] : "auto";
        auto device_index = std::numeric_limits<std::uint32_t>::max();
        if (device != "auto") {
            const auto index = parse_size(argv[9], "GPU device index");
            if (index >= std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("GPU device index is too large");
            device_index = static_cast<std::uint32_t>(index);
        }
        const std::string_view mode = argc > 10 ? argv[10] : "subgroup";
        if (mode != "subgroup" && mode != "workgroup")
            throw std::invalid_argument("GPU mode must be subgroup or workgroup");
        const std::string_view weight_memory = argc > 11 ? argv[11] : "auto";
        if (weight_memory != "auto" && weight_memory != "local" && weight_memory != "cached")
            throw std::invalid_argument("GPU weight memory must be auto, local, or cached");
        const std::string_view model_cache = argc > 12 ? argv[12] : "auto";
        if (model_cache != "auto" && model_cache != "off")
            throw std::invalid_argument("GPU model cache must be auto or off");
        const std::string_view vectorized_q4 = argc > 13 ? argv[13] : "true";
        if (vectorized_q4 != "true" && vectorized_q4 != "false")
            throw std::invalid_argument("GPU vectorized Q4 path must be true or false");

        const auto load_start = std::chrono::steady_clock::now();
        pokitlms::model::Qwen35Runner runner(argv[1], kv_window, kv_precision, io_threads);
#ifdef POKITLMS_USE_VULKAN
        std::shared_ptr<pokitlms::gpu::VulkanLinearBackend> vulkan;
        if (backend == "vulkan") {
            pokitlms::gpu::VulkanLinearOptions options;
            options.tile_bytes = tile_mib * 1024U * 1024U;
            options.device_index = device_index;
            options.use_subgroups = mode == "subgroup";
            options.use_vectorized_q4_k = vectorized_q4 == "true";
            if (model_cache == "off")
                options.model_cache_mode = pokitlms::gpu::VulkanModelCacheMode::Disabled;
            if (weight_memory == "local")
                options.weight_memory_mode = pokitlms::gpu::VulkanWeightMemoryMode::Local;
            else if (weight_memory == "cached")
                options.weight_memory_mode = pokitlms::gpu::VulkanWeightMemoryMode::Cached;
            vulkan = std::make_shared<pokitlms::gpu::VulkanLinearBackend>(options);
            runner.set_linear_backend(vulkan);
            std::cerr << "gpu_device=" << vulkan->device_name() << '\n'
                      << "gpu_tile_bytes=" << options.tile_bytes << '\n'
                      << "gpu_weight_memory_mode=" << weight_memory << '\n'
                      << "gpu_model_cache_mode=" << model_cache << '\n'
                      << "gpu_vectorized_q4_k=" << (options.use_vectorized_q4_k ? "true" : "false") << '\n'
                      << "gpu_weight_memory_flags=" << vulkan->stats().weight_memory_flags << '\n';
        }
#else
        (void)device_index;
        if (backend == "vulkan")
            throw std::runtime_error("Vulkan support was not compiled into this build");
#endif
        std::cerr << "backend=" << backend << '\n';
        const auto load_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - load_start).count());
        pokitlms::model::Qwen35GenerationStats stats;
        const auto bytes_before = runner.bytes_read_from_disk();
        std::vector<std::uint32_t> token_ids;
        const auto progress = [&](const pokitlms::model::Qwen35ProgressEvent& event) {
            using Phase = pokitlms::model::Qwen35ProgressEvent::Phase;
            const char* phase = "decode_forward";
            if (event.phase == Phase::PrefillToken) phase = "prefill_token";
            if (event.phase == Phase::GeneratedToken) {
                phase = "generated_token";
                token_ids.push_back(event.token_id);
            }
            std::cerr << "event=step phase=" << phase << " index=" << event.index
                      << " token_id=" << event.token_id
                      << " step_ms=" << milliseconds(event.elapsed_ns);
#if defined(__linux__)
            log_process_telemetry(process_telemetry());
#endif
            std::cerr << '\n';
#ifdef POKITLMS_USE_VULKAN
            if (vulkan && event.phase != Phase::GeneratedToken) {
                const auto gpu = vulkan->stats();
                std::cerr << "event=gpu_totals calls=" << gpu.linear_calls
                          << " dispatches=" << gpu.dispatches
                          << " vectorized_q4_k_calls=" << gpu.vectorized_q4_k_calls
                          << " q4_k_dispatches=" << gpu.q4_k_dispatches
                          << " q5_k_dispatches=" << gpu.q5_k_dispatches
                          << " q6_k_dispatches=" << gpu.q6_k_dispatches
                          << " q8_0_dispatches=" << gpu.q8_0_dispatches
                          << " other_dispatches=" << gpu.other_dispatches
                          << " weight_bytes=" << gpu.weight_bytes
                          << " read_ms=" << milliseconds(gpu.read_time_ns)
                          << " wait_ms=" << milliseconds(gpu.wait_time_ns)
                          << " gpu_ms=" << milliseconds(gpu.gpu_time_ns)
                          << " gpu_q4_k_ms=" << milliseconds(gpu.q4_k_gpu_time_ns)
                          << " gpu_q5_k_ms=" << milliseconds(gpu.q5_k_gpu_time_ns)
                          << " gpu_q6_k_ms=" << milliseconds(gpu.q6_k_gpu_time_ns)
                          << " gpu_q8_0_ms=" << milliseconds(gpu.q8_0_gpu_time_ns)
                          << " gpu_other_ms=" << milliseconds(gpu.other_gpu_time_ns)
                          << " pipeline_ms=" << milliseconds(gpu.pipeline_time_ns)
                          << " cache_ms=" << milliseconds(gpu.cache_time_ns)
                          << " model_cache_active=" << gpu.model_cache_active
                          << " model_cache_uploaded_bytes=" << gpu.model_cache_uploaded_bytes
                          << " model_cache_upload_ms=" << milliseconds(gpu.model_cache_upload_time_ns)
                          << " allocated_bytes=" << gpu.allocated_bytes << '\n';
            }
#endif
        };
#if defined(__linux__)
        const auto telemetry_before = process_telemetry();
#endif
        std::string response;
        try {
            response = runner.generate_chat(argv[2], max_new_tokens, &stats, progress);
        } catch (...) {
#ifdef POKITLMS_USE_VULKAN
            if (vulkan) {
                const auto gpu = vulkan->stats();
                std::cerr << "gpu_error_state model_cache_active=" << gpu.model_cache_active
                          << " model_cache_capacity_bytes=" << gpu.model_cache_capacity_bytes
                          << " model_cache_uploaded_bytes=" << gpu.model_cache_uploaded_bytes
                          << " model_cache_upload_ms=" << milliseconds(gpu.model_cache_upload_time_ns)
                          << " calls=" << gpu.linear_calls << " dispatches=" << gpu.dispatches << '\n';
            }
#endif
            throw;
        }
#if defined(__linux__)
        const auto telemetry_after = process_telemetry();
#endif
        const auto expert_cache = runner.expert_cache_stats();
#ifdef POKITLMS_USE_VULKAN
        const auto gpu_stats = vulkan ? vulkan->stats() : pokitlms::gpu::VulkanLinearStats{};
#endif
        std::cout << response << '\n';
        std::cerr << "model_load_ms=" << milliseconds(load_ns) << '\n'
                  << "generated_tokens=" << stats.generated_tokens << '\n'
                  << "first_generated_token_id="
                  << (stats.has_generated_token ? stats.first_generated_token_id : 0) << '\n'
                  << "prompt_tokens=" << stats.prompt_tokens << '\n'
                  << "prefill_ms=" << milliseconds(stats.prefill_time_ns) << '\n'
                  << "decode_ms=" << milliseconds(stats.decode_time_ns) << '\n'
                  << "decode_forward_tokens=" << stats.decode_forward_tokens << '\n'
                  << "prefill_tokens_per_second="
                  << tokens_per_second(stats.prompt_tokens, stats.prefill_time_ns) << '\n'
                  << "decode_tokens_per_second="
                  << tokens_per_second(stats.decode_forward_tokens, stats.decode_time_ns) << '\n'
                  << "model_bytes_read=" << runner.bytes_read_from_disk() - bytes_before << '\n'
                  << "kv_cache_storage_bytes=" << runner.kv_cache_storage_bytes() << '\n'
                  << "recurrent_state_storage_bytes="
                  << runner.recurrent_state_storage_bytes() << '\n'
                  << "expert_cache_capacity_bytes=" << expert_cache.capacity_bytes << '\n'
                  << "expert_cache_resident_bytes=" << expert_cache.resident_bytes << '\n'
                  << "expert_cache_bytes_read=" << expert_cache.bytes_read << '\n'
                  << "expert_read_operations=" << expert_cache.read_operations << '\n'
                  << "expert_read_ms=" << milliseconds(expert_cache.read_time_ns) << '\n'
                  << "expert_cache_hits=" << expert_cache.hits << '\n'
                  << "expert_cache_misses=" << expert_cache.misses << '\n';
#ifdef POKITLMS_USE_VULKAN
        if (vulkan) {
            std::cerr << "gpu_model_cache_active=" << gpu_stats.model_cache_active << '\n'
                      << "gpu_vectorized_q4_k_calls=" << gpu_stats.vectorized_q4_k_calls << '\n'
                      << "gpu_q4_k_dispatches=" << gpu_stats.q4_k_dispatches << '\n'
                      << "gpu_q5_k_dispatches=" << gpu_stats.q5_k_dispatches << '\n'
                      << "gpu_q6_k_dispatches=" << gpu_stats.q6_k_dispatches << '\n'
                      << "gpu_q8_0_dispatches=" << gpu_stats.q8_0_dispatches << '\n'
                      << "gpu_other_dispatches=" << gpu_stats.other_dispatches << '\n'
                      << "gpu_q4_k_ms=" << milliseconds(gpu_stats.q4_k_gpu_time_ns) << '\n'
                      << "gpu_q5_k_ms=" << milliseconds(gpu_stats.q5_k_gpu_time_ns) << '\n'
                      << "gpu_q6_k_ms=" << milliseconds(gpu_stats.q6_k_gpu_time_ns) << '\n'
                      << "gpu_q8_0_ms=" << milliseconds(gpu_stats.q8_0_gpu_time_ns) << '\n'
                      << "gpu_other_ms=" << milliseconds(gpu_stats.other_gpu_time_ns) << '\n'
                      << "gpu_model_cache_capacity_bytes=" << gpu_stats.model_cache_capacity_bytes << '\n'
                      << "gpu_model_cache_uploaded_bytes=" << gpu_stats.model_cache_uploaded_bytes << '\n'
                      << "gpu_model_cache_upload_ms=" << milliseconds(gpu_stats.model_cache_upload_time_ns) << '\n';
        }
#endif
#if defined(__linux__)
        // Process storage I/O is distinct from requested model bytes: cache hits
        // increase rchar without requiring a physical storage read.
        if (telemetry_before.read_bytes && telemetry_after.read_bytes &&
            *telemetry_after.read_bytes >= *telemetry_before.read_bytes) {
            std::cerr << "process_physical_read_bytes="
                      << *telemetry_after.read_bytes - *telemetry_before.read_bytes << '\n';
        }
        if (telemetry_before.rchar && telemetry_after.rchar &&
            *telemetry_after.rchar >= *telemetry_before.rchar) {
            std::cerr << "process_rchar_bytes="
                      << *telemetry_after.rchar - *telemetry_before.rchar << '\n';
        }
#endif
        std::cerr << "generated_token_ids=";
        for (std::size_t i = 0; i < token_ids.size(); ++i) {
            if (i) std::cerr << ',';
            std::cerr << token_ids[i];
        }
        std::cerr << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen35-bench: " << error.what() << '\n';
        return 1;
    }
}
