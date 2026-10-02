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
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
