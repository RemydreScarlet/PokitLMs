#include "pokitlms/model/qwen35_runner.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#ifndef POKITLMS_QWEN35_SMOKE_MODEL
#error POKITLMS_QWEN35_SMOKE_MODEL must point to the tiny test GGUF
#endif

int main() {
    try {
        pokitlms::model::Qwen35Runner runner(POKITLMS_QWEN35_SMOKE_MODEL);
        pokitlms::model::Qwen35GenerationStats stats;
        const auto answer = runner.generate_chat("hi", 3, &stats);
        if (answer != "aaa") throw std::runtime_error("unexpected generated text: " + answer);
        if (stats.prompt_tokens == 0 || stats.generated_tokens != 3) {
            throw std::runtime_error("generation statistics do not describe the smoke prompt");
        }
        if (runner.bytes_read_from_disk() == 0) {
            throw std::runtime_error("model weights were not read from the GGUF file");
        }
        std::vector<pokitlms::model::Qwen35ProgressEvent> events;
        const auto traced_answer = runner.generate_chat("hi", 3, &stats,
            [&events](const auto& event) { events.push_back(event); });
        std::size_t prompt_events = 0, generated_events = 0, decode_events = 0;
        for (const auto& event : events) {
            using Phase = pokitlms::model::Qwen35ProgressEvent::Phase;
            if (event.phase == Phase::PrefillToken) {
                if (event.index != ++prompt_events) throw std::runtime_error("unordered prompt progress");
            } else if (event.phase == Phase::GeneratedToken) {
                if (event.index != ++generated_events ||
                    event.token_id != stats.first_generated_token_id) {
                    throw std::runtime_error("unexpected generated-token progress");
                }
            } else if (event.index != ++decode_events) {
                throw std::runtime_error("unordered decode progress");
            }
        }
        if (traced_answer != answer || prompt_events != stats.prompt_tokens ||
            generated_events != 3 || decode_events != 2 || stats.decode_forward_tokens != 2) {
            throw std::runtime_error("progress callback changed output or omitted steps");
        }
        if (runner.config().mixture_of_experts) {
            for (std::size_t io_threads = 1; io_threads <= 4; ++io_threads) {
                pokitlms::model::Qwen35Runner tuned(
                    POKITLMS_QWEN35_SMOKE_MODEL, 0,
                    pokitlms::KvCachePrecision::Float16, io_threads);
                const auto tuned_answer = tuned.generate_chat("hi", 3);
                if (tuned_answer != answer) {
                    throw std::runtime_error("expert I/O worker count changed greedy output");
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
