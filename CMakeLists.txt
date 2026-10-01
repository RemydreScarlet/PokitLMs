cmake_minimum_required(VERSION 3.24)

project(PokitLMs VERSION 0.3.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(POKITLMS_BUILD_TESTS "Build PokitLMs tests" ON)
option(POKITLMS_BUILD_BENCHMARK "Build the model-backed inference benchmark CLI" ON)

add_library(pokitlms
    src/tensor.cpp
    src/rms_norm.cpp
    src/linear.cpp
    src/quantized_linear.cpp
    src/feed_forward.cpp
    src/rope.cpp
    src/expert_store.cpp
    src/model_file.cpp
    src/gguf_reader.cpp
    src/tensor_reader.cpp
    src/tensor_linear.cpp
    src/qwen3_moe_runner.cpp
    src/architecture.cpp
    src/kv_cache.cpp
    src/qwen3_moe_index.cpp
    src/qwen_bpe_tokenizer.cpp
    src/c_api.cpp
)

add_library(PokitLMs::pokitlms ALIAS pokitlms)

target_include_directories(pokitlms
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(pokitlms PUBLIC cxx_std_20)
target_link_libraries(pokitlms PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(pokitlms PRIVATE /W4)
else()
    target_compile_options(pokitlms PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(POKITLMS_BUILD_TESTS)
    enable_testing()
    add_executable(pokitlms_tests tests/smoke_test.cpp)
    target_link_libraries(pokitlms_tests PRIVATE PokitLMs::pokitlms)
    add_test(NAME pokitlms_smoke COMMAND pokitlms_tests)
endif()

if(POKITLMS_BUILD_BENCHMARK)
    add_executable(pokitlms-bench tools/benchmark.cpp)
    target_link_libraries(pokitlms-bench PRIVATE PokitLMs::pokitlms)
endif()
