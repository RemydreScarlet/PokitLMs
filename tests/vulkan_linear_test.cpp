#include "vulkan_linear_cases.hpp"
#include "pokitlms/model/qwen35_runner.hpp"

#include <iostream>

int main() {
    try {
        std::cout << run_vulkan_linear_checks();
        for (const bool allow_model_cache : {true, false}) {
            pokitlms::gpu::VulkanLinearOptions options;
            if (!allow_model_cache) {
                options.model_cache_mode = pokitlms::gpu::VulkanModelCacheMode::Disabled;
            }
            auto backend = std::make_shared<pokitlms::gpu::VulkanLinearBackend>(options);
            for (const char* path : {POKITLMS_QWEN35_SMOKE_MODEL, POKITLMS_QWEN35_MOE_SMOKE_MODEL}) {
                pokitlms::model::Qwen35Runner cpu(path), gpu(path);
                gpu.set_linear_backend(backend);
                for (const auto& prompt : {"hi", "hello", "again"}) {
                    const auto expected = cpu.generate_chat(prompt, 5);
                    if (gpu.generate_chat(prompt, 5) != expected) {
                        throw std::runtime_error("Vulkan Dense/MoE output differs");
                    }
                }
            }
            const auto stats = backend->stats();
            if (!stats.linear_calls) throw std::runtime_error("generation did not use Vulkan");
            if (!allow_model_cache && (stats.model_cache_active || stats.model_cache_uploaded_bytes)) {
                throw std::runtime_error("disabled Vulkan model cache became active");
            }
            if (stats.model_cache_active && !stats.model_cache_uploaded_bytes) {
                throw std::runtime_error("Vulkan model cache has no uploaded bytes");
            }
            std::cout << "model_cache_mode=" << (allow_model_cache ? "auto" : "off")
                      << " active=" << stats.model_cache_active
                      << " uploaded_bytes=" << stats.model_cache_uploaded_bytes << '\n';
        }
        std::cout << "Dense/MoE token parity PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
