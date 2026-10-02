#include <jni.h>
#include <android/log.h>

#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/model/qwen3_moe_runner.hpp"
#include "pokitlms/model/qwen35_runner.hpp"
#ifdef POKITLMS_USE_VULKAN
#include "pokitlms/gpu/vulkan_linear.hpp"
#endif
#ifdef POKITLMS_GPU_DIAGNOSTICS
#include "vulkan_linear_cases.hpp"
#endif

#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <variant>
#include <vector>

namespace {
struct Engine {
    Engine(const std::string& path, std::size_t expert_io_threads, bool use_vulkan,
           [[maybe_unused]] std::size_t gpu_tile_mib, [[maybe_unused]] bool gpu_subgroups) {
        pokitlms::model::GgufReader gguf(path);
        const auto it = gguf.metadata().find("general.architecture");
        if (it == gguf.metadata().end() || !std::holds_alternative<std::string>(it->second.value))
            throw std::runtime_error("GGUF has no valid general.architecture metadata");
        const auto& architecture = std::get<std::string>(it->second.value);
        if (architecture == "qwen3moe")
            moe = std::make_unique<pokitlms::model::Qwen3MoeRunner>(
                path, 128U * 1024U * 1024U, 0, pokitlms::KvCachePrecision::Float16,
                expert_io_threads);
        else if (architecture == "qwen35" || architecture == "qwen35moe")
            dense = std::make_unique<pokitlms::model::Qwen35Runner>(
                path, 0, pokitlms::KvCachePrecision::Float16, expert_io_threads);
        else throw std::runtime_error("Unsupported GGUF architecture: " + architecture);
        if (use_vulkan && dense) {
#ifdef POKITLMS_USE_VULKAN
            pokitlms::gpu::VulkanLinearOptions options;
            options.tile_bytes = gpu_tile_mib * 1024U * 1024U;
            options.use_subgroups = gpu_subgroups;
            vulkan = std::make_shared<pokitlms::gpu::VulkanLinearBackend>(options);
            dense->set_linear_backend(vulkan);
            __android_log_print(ANDROID_LOG_INFO, "PokitLMsAB",
                "event=backend_init backend=vulkan device=%s tile_bytes=%zu weight_memory_flags=%u",
                vulkan->device_name().c_str(), options.tile_bytes, vulkan->stats().weight_memory_flags);
#else
            throw std::runtime_error("Vulkan support was not compiled into this build");
#endif
        } else {
            __android_log_print(ANDROID_LOG_INFO, "PokitLMsAB", "event=backend_init backend=cpu");
        }
    }
    std::unique_ptr<pokitlms::model::Qwen3MoeRunner> moe;
    std::unique_ptr<pokitlms::model::Qwen35Runner> dense;
#ifdef POKITLMS_USE_VULKAN
    std::shared_ptr<pokitlms::gpu::VulkanLinearBackend> vulkan;
#endif
};

jstring java_string(JNIEnv* env, const std::string& value) {
    std::vector<jchar> utf16;
    for (std::size_t i = 0; i < value.size();) {
        const auto first = static_cast<unsigned char>(value[i++]);
        std::uint32_t cp = first;
        if ((first & 0xe0U) == 0xc0U && i < value.size()) {
            const auto second = static_cast<unsigned char>(value[i++]);
            cp = ((first & 0x1fU) << 6U) | (second & 0x3fU);
        } else if ((first & 0xf0U) == 0xe0U && i + 1 < value.size()) {
            const auto second = static_cast<unsigned char>(value[i++]);
            const auto third = static_cast<unsigned char>(value[i++]);
            cp = ((first & 0x0fU) << 12U) | ((second & 0x3fU) << 6U) | (third & 0x3fU);
        } else if ((first & 0xf8U) == 0xf0U && i + 2 < value.size()) {
            const auto second = static_cast<unsigned char>(value[i++]);
            const auto third = static_cast<unsigned char>(value[i++]);
            const auto fourth = static_cast<unsigned char>(value[i++]);
            cp = ((first & 0x07U) << 18U) | ((second & 0x3fU) << 12U) |
                 ((third & 0x3fU) << 6U) | (fourth & 0x3fU);
        }
        if (cp <= 0xffffU) utf16.push_back(static_cast<jchar>(cp));
        else {
            cp -= 0x10000U;
            utf16.push_back(static_cast<jchar>(0xd800U + (cp >> 10U)));
            utf16.push_back(static_cast<jchar>(0xdc00U + (cp & 0x3ffU)));
        }
    }
    return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

std::string utf8_string(JNIEnv* env, jstring value) {
    const jsize length = env->GetStringLength(value);
    const jchar* chars = env->GetStringChars(value, nullptr);
    if (!chars) throw std::runtime_error("Could not read prompt text");
    std::string result;
    result.reserve(static_cast<std::size_t>(length) * 3);
    for (jsize i = 0; i < length; ++i) {
        std::uint32_t cp = chars[i];
        if (cp >= 0xd800U && cp <= 0xdbffU && i + 1 < length &&
            chars[i + 1] >= 0xdc00U && chars[i + 1] <= 0xdfffU) {
            cp = 0x10000U + ((cp - 0xd800U) << 10U) + (chars[++i] - 0xdc00U);
        }
        if (cp <= 0x7fU) result.push_back(static_cast<char>(cp));
        else if (cp <= 0x7ffU) {
            result.push_back(static_cast<char>(0xc0U | (cp >> 6U)));
            result.push_back(static_cast<char>(0x80U | (cp & 0x3fU)));
        } else if (cp <= 0xffffU) {
            result.push_back(static_cast<char>(0xe0U | (cp >> 12U)));
            result.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | (cp & 0x3fU)));
        } else {
            result.push_back(static_cast<char>(0xf0U | (cp >> 18U)));
            result.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3fU)));
            result.push_back(static_cast<char>(0x80U | (cp & 0x3fU)));
        }
    }
    env->ReleaseStringChars(value, chars);
    return result;
}

void throw_java(JNIEnv* env, const char* type, const std::string& message) {
    jclass cls = env->FindClass(type);
    if (cls) env->ThrowNew(cls, message.c_str());
}

std::uint64_t proc_kb_value(const char* path, const char* field) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream parsed(line);
        std::string name;
        std::uint64_t value = 0;
        if (parsed >> name >> value && name == field) return value;
    }
    return 0;
}

double timeval_ms(const timeval& value) {
    return static_cast<double>(value.tv_sec) * 1000.0 +
           static_cast<double>(value.tv_usec) / 1000.0;
}

const char* progress_phase(pokitlms::model::Qwen35ProgressEvent::Phase phase) {
    using Phase = pokitlms::model::Qwen35ProgressEvent::Phase;
    switch (phase) {
        case Phase::PrefillToken: return "prefill_token";
        case Phase::GeneratedToken: return "generated_token";
        case Phase::DecodeForward: return "decode_forward";
    }
    return "unknown";
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_pokit_pokitlms_NativeModelBridge_load(JNIEnv* env, jobject, jint fd,
                                               jint expert_io_threads, jboolean use_vulkan,
                                               jint gpu_tile_mib, jboolean gpu_subgroups) {
    try {
        if (expert_io_threads < 1 || expert_io_threads > 4)
            throw std::invalid_argument("expert I/O threads must be between 1 and 4");
        if (gpu_tile_mib < 1 || gpu_tile_mib > 128)
            throw std::invalid_argument("GPU tile MiB must be between 1 and 128");
        const std::string path = "/proc/self/fd/" + std::to_string(fd);
        return reinterpret_cast<jlong>(new Engine(path,
            static_cast<std::size_t>(expert_io_threads), use_vulkan == JNI_TRUE,
            static_cast<std::size_t>(gpu_tile_mib), gpu_subgroups == JNI_TRUE));
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalArgumentException", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_pokit_pokitlms_NativeModelBridge_generate(JNIEnv* env, jobject, jlong handle,
                                                    jstring prompt, jint max_tokens) {
    try {
        auto* engine = reinterpret_cast<Engine*>(handle);
        if (!engine) throw std::runtime_error("Model is not loaded");
        const std::string message = utf8_string(env, prompt);
        if (engine->moe) {
            const auto answer = engine->moe->generate_chat(message, max_tokens);
            return java_string(env, answer);
        }

        pokitlms::model::Qwen35GenerationStats stats;
        const pokitlms::model::Qwen35ProgressCallback progress =
            [engine](const pokitlms::model::Qwen35ProgressEvent& event) {
                rusage usage{};
                getrusage(RUSAGE_SELF, &usage);
                const auto rss_kb = proc_kb_value("/proc/self/status", "VmRSS:");
                const auto swap_kb = proc_kb_value("/proc/self/status", "VmSwap:");
                const auto vm_kb = proc_kb_value("/proc/self/status", "VmSize:");
                const auto available_kb = proc_kb_value("/proc/meminfo", "MemAvailable:");
                const auto swap_free_kb = proc_kb_value("/proc/meminfo", "SwapFree:");
                const auto read_bytes = proc_kb_value("/proc/self/io", "read_bytes:");
                const auto rchar = proc_kb_value("/proc/self/io", "rchar:");
                __android_log_print(ANDROID_LOG_INFO, "PokitLMsAB",
                    "event=step phase=%s index=%zu token_id=%u step_ms=%.3f "
                    "cpu_user_ms=%.3f cpu_sys_ms=%.3f rss_kb=%llu swap_kb=%llu "
                    "vm_kb=%llu mem_available_kb=%llu swap_free_kb=%llu read_bytes=%llu rchar=%llu",
                    progress_phase(event.phase), event.index, event.token_id,
                    static_cast<double>(event.elapsed_ns) / 1'000'000.0,
                    timeval_ms(usage.ru_utime), timeval_ms(usage.ru_stime),
                    static_cast<unsigned long long>(rss_kb),
                    static_cast<unsigned long long>(swap_kb),
                    static_cast<unsigned long long>(vm_kb),
                    static_cast<unsigned long long>(available_kb),
                    static_cast<unsigned long long>(swap_free_kb),
                    static_cast<unsigned long long>(read_bytes),
                    static_cast<unsigned long long>(rchar));
#ifdef POKITLMS_USE_VULKAN
                if (engine->vulkan && event.phase != pokitlms::model::Qwen35ProgressEvent::Phase::GeneratedToken) {
                    const auto gpu = engine->vulkan->stats();
                    __android_log_print(ANDROID_LOG_INFO, "PokitLMsAB",
                        "event=gpu_totals calls=%llu dispatches=%llu weight_bytes=%llu "
                        "read_ms=%.3f wait_ms=%.3f gpu_ms=%.3f pipeline_ms=%.3f cache_ms=%.3f allocated_bytes=%llu",
                        static_cast<unsigned long long>(gpu.linear_calls),
                        static_cast<unsigned long long>(gpu.dispatches),
                        static_cast<unsigned long long>(gpu.weight_bytes),
                        static_cast<double>(gpu.read_time_ns) / 1'000'000.0,
                        static_cast<double>(gpu.wait_time_ns) / 1'000'000.0,
                        static_cast<double>(gpu.gpu_time_ns) / 1'000'000.0,
                        static_cast<double>(gpu.pipeline_time_ns) / 1'000'000.0,
                        static_cast<double>(gpu.cache_time_ns) / 1'000'000.0,
                        static_cast<unsigned long long>(gpu.allocated_bytes));
                }
#endif
            };
        __android_log_print(ANDROID_LOG_INFO, "PokitLMsAB",
                            "event=generate_start arch=qwen35 max_tokens=%d prompt_chars=%zu",
                            max_tokens, message.size());
        const auto answer = engine->dense->generate_chat(message, max_tokens, &stats, progress);
        __android_log_print(ANDROID_LOG_INFO, "PokitLMsAB",
                            "event=generate_summary prompt_tokens=%zu generated_tokens=%zu "
                            "decode_forward_tokens=%zu prefill_ms=%.3f decode_ms=%.3f "
                            "first_token_id=%u has_token=%d",
                            stats.prompt_tokens, stats.generated_tokens, stats.decode_forward_tokens,
                            static_cast<double>(stats.prefill_time_ns) / 1'000'000.0,
                            static_cast<double>(stats.decode_time_ns) / 1'000'000.0,
                            stats.first_generated_token_id, stats.has_generated_token ? 1 : 0);
        return java_string(env, answer);
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_org_pokit_pokitlms_NativeModelBridge_close(JNIEnv*, jobject, jlong handle) {
    delete reinterpret_cast<Engine*>(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_pokit_pokitlms_NativeModelBridge_verifyVulkan(JNIEnv* env, jobject) {
    try {
#ifdef POKITLMS_GPU_DIAGNOSTICS
        return java_string(env, run_vulkan_linear_checks());
#else
        throw std::runtime_error("GPU diagnostics require the benchmark build");
#endif
    } catch (const std::exception& error) {
        throw_java(env, "java/lang/RuntimeException", error.what());
        return nullptr;
    }
}
