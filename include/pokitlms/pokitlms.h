#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POKITLMS_VERSION_MAJOR 0
#define POKITLMS_VERSION_MINOR 3
#define POKITLMS_VERSION_PATCH 0

const char* pokitlms_version(void);

typedef struct pokitlms_model pokitlms_model;

typedef enum pokitlms_status {
    POKITLMS_STATUS_OK = 0,
    POKITLMS_STATUS_INVALID_ARGUMENT = 1,
    POKITLMS_STATUS_RUNTIME_ERROR = 2,
    POKITLMS_STATUS_BUFFER_TOO_SMALL = 3,
    POKITLMS_STATUS_INTERNAL_ERROR = 4
} pokitlms_status;

typedef struct pokitlms_generation_options {
    float temperature;            // 0 selects greedy decoding.
    size_t top_k;                 // 0 disables top-k filtering.
    float top_p;                  // Nucleus threshold in (0, 1].
    float repetition_penalty;     // 1 disables the penalty.
    uint64_t seed;
} pokitlms_generation_options;

void pokitlms_generation_options_init(pokitlms_generation_options* options);

typedef struct pokitlms_expert_cache_stats {
    size_t capacity_bytes;
    size_t resident_bytes;
    uint64_t bytes_read;
    uint64_t read_operations; // Expert-cache misses that performed a file read.
    uint64_t read_time_ns;    // Cumulative read-call time, including cache advice.
    uint64_t hits;
    uint64_t misses;
} pokitlms_expert_cache_stats;

typedef struct pokitlms_chat_message {
    const char* role;    // "system", "user", or "assistant".
    const char* content; // UTF-8 message content.
} pokitlms_chat_message;

// Creates a single-sequence Qwen3-MoE model handle. context_capacity sets the
// resident KV window (older positions roll out); zero selects the mobile default.
// Error text, when provided, is truncated and NUL-terminated.
pokitlms_status pokitlms_qwen3moe_create(
    const char* model_path, size_t expert_cache_budget_bytes, size_t context_capacity,
    pokitlms_model** out_model, char* error_buffer, size_t error_capacity);
void pokitlms_model_destroy(pokitlms_model* model);

// Generates UTF-8 output from a raw prompt; supply any chat-template markers in
// the prompt. Null options selects defaults; zero top_p/repetition_penalty also
// select their defaults. output_length excludes NUL. Capacity must fit length + 1.
// BUFFER_TOO_SMALL still fills output_length, so the caller can retry with more space.
// Calls on one handle must be serialized.
pokitlms_status pokitlms_model_generate_text(
    pokitlms_model* model, const char* prompt, size_t max_new_tokens,
    const pokitlms_generation_options* options,
    char* output, size_t output_capacity, size_t* output_length,
    char* error_buffer, size_t error_capacity);

// Generates an assistant reply to one UTF-8 user message using Qwen3's
// standard text chat format. Buffer and error semantics match generate_text.
pokitlms_status pokitlms_model_generate_chat(
    pokitlms_model* model, const char* user_message, size_t max_new_tokens,
    const pokitlms_generation_options* options,
    char* output, size_t output_capacity, size_t* output_length,
    char* error_buffer, size_t error_capacity);

// Generates a reply from a system/user/assistant history. Messages must
// alternate user and assistant after an optional leading system message, and
// the final message must be from the user. The history is re-prefilled each call.
pokitlms_status pokitlms_model_generate_chat_history(
    pokitlms_model* model, const pokitlms_chat_message* messages, size_t message_count,
    size_t max_new_tokens, const pokitlms_generation_options* options,
    char* output, size_t output_capacity, size_t* output_length,
    char* error_buffer, size_t error_capacity);

pokitlms_status pokitlms_model_get_expert_cache_stats(
    const pokitlms_model* model, pokitlms_expert_cache_stats* out_stats);
uint64_t pokitlms_model_bytes_read_from_disk(const pokitlms_model* model);

#ifdef __cplusplus
}
#endif
