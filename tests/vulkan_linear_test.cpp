#include "vulkan_linear_cases.hpp"
#include "pokitlms/model/qwen35_runner.hpp"

#include <iostream>

int main() {
    try {
        std::cout << run_vulkan_linear_checks();
        auto backend = std::make_shared<pokitlms::gpu::VulkanLinearBackend>();
        for (const char* path : {POKITLMS_QWEN35_SMOKE_MODEL, POKITLMS_QWEN35_MOE_SMOKE_MODEL}) {
            pokitlms::model::Qwen35Runner cpu(path), gpu(path);
            gpu.set_linear_backend(backend);
            for (const auto& prompt : {"hi", "hello", "again"}) {
                const auto expected = cpu.generate_chat(prompt, 5);
                if (gpu.generate_chat(prompt, 5) != expected) throw std::runtime_error("Vulkan Dense/MoE output differs");
            }
        }
        if (!backend->stats().linear_calls) throw std::runtime_error("generation did not use Vulkan");
        std::cout << "Dense/MoE token parity PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
