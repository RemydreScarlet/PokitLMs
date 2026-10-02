#include "pokitlms/model/qwen3_moe_runner.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#ifndef POKITLMS_QWEN3MOE_SMOKE_MODEL
#error POKITLMS_QWEN3MOE_SMOKE_MODEL must point to the tiny test GGUF
#endif

int main() {
    try {
        pokitlms::model::Qwen3MoeRunner runner(
            POKITLMS_QWEN3MOE_SMOKE_MODEL, 1024U * 1024U, 512,
            pokitlms::KvCachePrecision::Float16, 2);
        const auto answer = runner.generate_chat("hi", 3);
        if (answer != "aaa") throw std::runtime_error("unexpected generated text: " + answer);
        if (runner.bytes_read_from_disk() == 0) {
            throw std::runtime_error("model weights were not read from the GGUF file");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
