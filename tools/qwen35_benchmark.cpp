#include "pokitlms/model/qwen35_runner.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

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

void usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " MODEL.gguf USER_MESSAGE [new_tokens=32] [kv_window=0] [kv_precision=fp16|q8]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
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

        const auto load_start = std::chrono::steady_clock::now();
        pokitlms::model::Qwen35Runner runner(argv[1], kv_window, kv_precision);
        const auto load_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - load_start).count());
        pokitlms::model::Qwen35GenerationStats stats;
        const auto bytes_before = runner.bytes_read_from_disk();
        const auto response = runner.generate_chat(argv[2], max_new_tokens, &stats);
        std::cout << response << '\n';
        std::cerr << "model_load_ms=" << milliseconds(load_ns) << '\n'
                  << "generated_tokens=" << stats.generated_tokens << '\n'
                  << "prompt_tokens=" << stats.prompt_tokens << '\n'
                  << "prefill_ms=" << milliseconds(stats.prefill_time_ns) << '\n'
                  << "decode_ms=" << milliseconds(stats.decode_time_ns) << '\n'
                  << "prefill_tokens_per_second="
                  << tokens_per_second(stats.prompt_tokens, stats.prefill_time_ns) << '\n'
                  << "decode_tokens_per_second="
                  << tokens_per_second(stats.generated_tokens, stats.decode_time_ns) << '\n'
                  << "model_bytes_read=" << runner.bytes_read_from_disk() - bytes_before << '\n'
                  << "kv_cache_storage_bytes=" << runner.kv_cache_storage_bytes() << '\n'
                  << "recurrent_state_storage_bytes="
                  << runner.recurrent_state_storage_bytes() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen35-bench: " << error.what() << '\n';
        return 1;
    }
}
