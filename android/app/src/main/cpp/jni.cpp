#include <jni.h>

#include "pokitlms/model/gguf_reader.hpp"
#include "pokitlms/model/qwen3_moe_runner.hpp"
#include "pokitlms/model/qwen35_runner.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
struct Engine {
    explicit Engine(const std::string& path) {
        pokitlms::model::GgufReader gguf(path);
        const auto it = gguf.metadata().find("general.architecture");
        if (it == gguf.metadata().end() || !std::holds_alternative<std::string>(it->second.value))
            throw std::runtime_error("GGUF has no valid general.architecture metadata");
        const auto& architecture = std::get<std::string>(it->second.value);
        if (architecture == "qwen3moe") moe = std::make_unique<pokitlms::model::Qwen3MoeRunner>(path);
        else if (architecture == "qwen35" || architecture == "qwen35moe")
            dense = std::make_unique<pokitlms::model::Qwen35Runner>(path);
        else throw std::runtime_error("Unsupported GGUF architecture: " + architecture);
    }
    std::unique_ptr<pokitlms::model::Qwen3MoeRunner> moe;
    std::unique_ptr<pokitlms::model::Qwen35Runner> dense;
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
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_pokit_pokitlms_NativeModelBridge_load(JNIEnv* env, jobject, jint fd) {
    try {
        const std::string path = "/proc/self/fd/" + std::to_string(fd);
        return reinterpret_cast<jlong>(new Engine(path));
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
        const auto answer = engine->moe ? engine->moe->generate_chat(message, max_tokens)
                                        : engine->dense->generate_chat(message, max_tokens);
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
