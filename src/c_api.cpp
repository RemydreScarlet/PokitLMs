#include "pokitlms/pokitlms.h"

#include "pokitlms/model/qwen3_moe_runner.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <utility>

struct pokitlms_model {
    explicit pokitlms_model(std::filesystem::path path, std::size_t cache_bytes,
                            std::size_t context, pokitlms::KvCachePrecision kv_precision,
                            std::size_t expert_io_threads)
        : runner(std::move(path), cache_bytes, context, kv_precision, expert_io_threads) {}
    pokitlms::model::Qwen3MoeRunner runner;
};

namespace {

void set_error(char* destination, std::size_t capacity, const char* message) noexcept {
    if (!destination || capacity == 0) return;
    const auto count = std::min(capacity - 1, std::strlen(message));
    std::memcpy(destination, message, count);
    destination[count] = '\0';
}

pokitlms::model::GenerationOptions convert_options(const pokitlms_generation_options* options) {
    if (!options) return {};
    return {options->temperature, options->top_k,
            options->top_p == 0.0F ? 1.0F : options->top_p,
            options->repetition_penalty == 0.0F ? 1.0F : options->repetition_penalty,
            options->seed};
}

}  // namespace

const char* pokitlms_version(void) {
#define POKITLMS_STRINGIFY_IMPL(value) #value
#define POKITLMS_STRINGIFY(value) POKITLMS_STRINGIFY_IMPL(value)
    return POKITLMS_STRINGIFY(POKITLMS_VERSION_MAJOR) "."
           POKITLMS_STRINGIFY(POKITLMS_VERSION_MINOR) "."
           POKITLMS_STRINGIFY(POKITLMS_VERSION_PATCH);
#undef POKITLMS_STRINGIFY
#undef POKITLMS_STRINGIFY_IMPL
}

void pokitlms_generation_options_init(pokitlms_generation_options* options) {
    if (!options) return;
    *options = {0.0F, 0, 1.0F, 1.0F, 0};
}

pokitlms_status pokitlms_qwen3moe_create(
    const char* model_path, size_t expert_cache_budget_bytes, size_t context_capacity,
    pokitlms_model** out_model, char* error_buffer, size_t error_capacity) {
    return pokitlms_qwen3moe_create_ex(model_path, expert_cache_budget_bytes, context_capacity,
                                       POKITLMS_KV_CACHE_FP16, out_model, error_buffer,
                                       error_capacity);
}

pokitlms_status pokitlms_qwen3moe_create_ex(
    const char* model_path, size_t expert_cache_budget_bytes, size_t context_capacity,
    pokitlms_kv_cache_precision kv_precision, pokitlms_model** out_model,
    char* error_buffer, size_t error_capacity) {
    return pokitlms_qwen3moe_create_ex_with_io_threads(
        model_path, expert_cache_budget_bytes, context_capacity, kv_precision, 3,
        out_model, error_buffer, error_capacity);
}

pokitlms_status pokitlms_qwen3moe_create_ex_with_io_threads(
    const char* model_path, size_t expert_cache_budget_bytes, size_t context_capacity,
    pokitlms_kv_cache_precision kv_precision, size_t expert_io_threads,
    pokitlms_model** out_model, char* error_buffer, size_t error_capacity) {
    if (out_model) *out_model = nullptr;
    if (!model_path || model_path[0] == '\0' || !out_model) {
        set_error(error_buffer, error_capacity, "model path and output handle are required");
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    }
    pokitlms::KvCachePrecision precision;
    if (kv_precision == POKITLMS_KV_CACHE_FP16) precision = pokitlms::KvCachePrecision::Float16;
    else if (kv_precision == POKITLMS_KV_CACHE_Q8_0) precision = pokitlms::KvCachePrecision::Q8_0;
    else {
        set_error(error_buffer, error_capacity, "unsupported KV cache precision");
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    }
    try {
        *out_model = new pokitlms_model(model_path, expert_cache_budget_bytes,
                                        context_capacity, precision, expert_io_threads);
        set_error(error_buffer, error_capacity, "");
        return POKITLMS_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(error_buffer, error_capacity, "unknown model creation failure");
        return POKITLMS_STATUS_INTERNAL_ERROR;
    }
}

void pokitlms_model_destroy(pokitlms_model* model) { delete model; }

pokitlms_status pokitlms_model_generate_text(
    pokitlms_model* model, const char* prompt, size_t max_new_tokens,
    const pokitlms_generation_options* options,
    char* output, size_t output_capacity, size_t* output_length,
    char* error_buffer, size_t error_capacity) {
    if (output_length) *output_length = 0;
    if (!model || !prompt || !output_length) {
        set_error(error_buffer, error_capacity, "model, prompt, and output length are required");
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    }
    try {
        const auto text = model->runner.generate_text(prompt, max_new_tokens, convert_options(options));
        *output_length = text.size();
        if (!output || output_capacity <= text.size()) {
            set_error(error_buffer, error_capacity, "output buffer must fit generated text and a NUL byte");
            return POKITLMS_STATUS_BUFFER_TOO_SMALL;
        }
        std::memcpy(output, text.data(), text.size());
        output[text.size()] = '\0';
        set_error(error_buffer, error_capacity, "");
        return POKITLMS_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(error_buffer, error_capacity, "unknown text generation failure");
        return POKITLMS_STATUS_INTERNAL_ERROR;
    }
}

pokitlms_status pokitlms_model_generate_chat(
    pokitlms_model* model, const char* user_message, size_t max_new_tokens,
    const pokitlms_generation_options* options,
    char* output, size_t output_capacity, size_t* output_length,
    char* error_buffer, size_t error_capacity) {
    if (output_length) *output_length = 0;
    if (!model || !user_message || !output_length) {
        set_error(error_buffer, error_capacity, "model, user message, and output length are required");
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    }
    try {
        const auto text = model->runner.generate_chat(user_message, max_new_tokens,
                                                       convert_options(options));
        *output_length = text.size();
        if (!output || output_capacity <= text.size()) {
            set_error(error_buffer, error_capacity, "output buffer must fit generated text and a NUL byte");
            return POKITLMS_STATUS_BUFFER_TOO_SMALL;
        }
        std::memcpy(output, text.data(), text.size());
        output[text.size()] = '\0';
        set_error(error_buffer, error_capacity, "");
        return POKITLMS_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(error_buffer, error_capacity, "unknown chat generation failure");
        return POKITLMS_STATUS_INTERNAL_ERROR;
    }
}

pokitlms_status pokitlms_model_generate_chat_history(
    pokitlms_model* model, const pokitlms_chat_message* messages, size_t message_count,
    size_t max_new_tokens, const pokitlms_generation_options* options,
    char* output, size_t output_capacity, size_t* output_length,
    char* error_buffer, size_t error_capacity) {
    if (output_length) *output_length = 0;
    if (!model || !messages || message_count == 0 || !output_length) {
        set_error(error_buffer, error_capacity, "model, chat messages, and output length are required");
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::vector<pokitlms::model::ChatMessage> history;
        history.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            if (!messages[i].role || !messages[i].content) {
                set_error(error_buffer, error_capacity, "chat message roles and contents cannot be null");
                return POKITLMS_STATUS_INVALID_ARGUMENT;
            }
            history.push_back({messages[i].role, messages[i].content});
        }
        const auto text = model->runner.generate_chat(history, max_new_tokens,
                                                       convert_options(options));
        *output_length = text.size();
        if (!output || output_capacity <= text.size()) {
            set_error(error_buffer, error_capacity, "output buffer must fit generated text and a NUL byte");
            return POKITLMS_STATUS_BUFFER_TOO_SMALL;
        }
        std::memcpy(output, text.data(), text.size());
        output[text.size()] = '\0';
        set_error(error_buffer, error_capacity, "");
        return POKITLMS_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        set_error(error_buffer, error_capacity, error.what());
        return POKITLMS_STATUS_RUNTIME_ERROR;
    } catch (...) {
        set_error(error_buffer, error_capacity, "unknown chat history generation failure");
        return POKITLMS_STATUS_INTERNAL_ERROR;
    }
}

pokitlms_status pokitlms_model_get_expert_cache_stats(
    const pokitlms_model* model, pokitlms_expert_cache_stats* out_stats) {
    if (!model || !out_stats) return POKITLMS_STATUS_INVALID_ARGUMENT;
    try {
        const auto stats = model->runner.expert_cache_stats();
        *out_stats = {stats.capacity_bytes, stats.resident_bytes, stats.bytes_read,
                      stats.read_operations, stats.read_time_ns, stats.hits, stats.misses};
        return POKITLMS_STATUS_OK;
    } catch (...) {
        return POKITLMS_STATUS_INTERNAL_ERROR;
    }
}

uint64_t pokitlms_model_bytes_read_from_disk(const pokitlms_model* model) {
    return model ? model->runner.bytes_read_from_disk() : 0;
}

size_t pokitlms_model_kv_cache_storage_bytes(const pokitlms_model* model) {
    return model ? model->runner.kv_cache_storage_bytes() : 0;
}
