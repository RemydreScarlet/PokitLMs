cmake_minimum_required(VERSION 3.22)

project(PokitLMs VERSION 0.4.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(POKITLMS_BUILD_TESTS "Build PokitLMs tests" ON)
option(POKITLMS_BUILD_BENCHMARK "Build the model-backed inference benchmark CLI" ON)
option(POKITLMS_BUILD_ANDROID_APP "Build the Android JNI bridge" OFF)

add_library(pokitlms
    src/tensor.cpp
    src/rms_norm.cpp
    src/linear.cpp
    src/quantized_linear.cpp
    src/feed_forward.cpp
    src/rope.cpp
    src/causal_conv1d.cpp
    src/gated_delta_net.cpp
    src/expert_store.cpp
    src/model_file.cpp
    src/gguf_reader.cpp
    src/tensor_reader.cpp
    src/tensor_linear.cpp
    src/qwen3_moe_runner.cpp
    src/architecture.cpp
    src/kv_cache.cpp
    src/reference_kv_cache.cpp
    src/attention.cpp
    src/qwen3_moe_index.cpp
    src/qwen35_index.cpp
    src/qwen35_runner.cpp
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
    add_executable(pokitlms_regression_tests tests/regression_test.cpp)
    target_link_libraries(pokitlms_regression_tests PRIVATE PokitLMs::pokitlms)
    add_test(NAME pokitlms_regressions COMMAND pokitlms_regression_tests)

    add_executable(pokitlms_qwen35_smoke_tests tests/qwen35_smoke_test.cpp)
    target_link_libraries(pokitlms_qwen35_smoke_tests PRIVATE PokitLMs::pokitlms)
    target_compile_definitions(pokitlms_qwen35_smoke_tests PRIVATE
        POKITLMS_QWEN35_SMOKE_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen35-smoke.gguf")
    add_test(NAME pokitlms_qwen35_smoke COMMAND pokitlms_qwen35_smoke_tests)

    add_executable(pokitlms_qwen35_moe_smoke_tests tests/qwen35_smoke_test.cpp)
    target_link_libraries(pokitlms_qwen35_moe_smoke_tests PRIVATE PokitLMs::pokitlms)
    target_compile_definitions(pokitlms_qwen35_moe_smoke_tests PRIVATE
        POKITLMS_QWEN35_SMOKE_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen35moe-smoke.gguf")
    add_test(NAME pokitlms_qwen35_moe_smoke COMMAND pokitlms_qwen35_moe_smoke_tests)

    add_executable(pokitlms_qwen3_moe_smoke_tests tests/qwen3_moe_smoke_test.cpp)
    target_link_libraries(pokitlms_qwen3_moe_smoke_tests PRIVATE PokitLMs::pokitlms)
    target_compile_definitions(pokitlms_qwen3_moe_smoke_tests PRIVATE
        POKITLMS_QWEN3MOE_SMOKE_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen3moe-smoke.gguf")
    add_test(NAME pokitlms_qwen3_moe_smoke COMMAND pokitlms_qwen3_moe_smoke_tests)
endif()

if(POKITLMS_BUILD_BENCHMARK)
    add_executable(pokitlms-bench tools/benchmark.cpp)
    target_link_libraries(pokitlms-bench PRIVATE PokitLMs::pokitlms)

    add_executable(pokitlms-qwen35-bench tools/qwen35_benchmark.cpp)
    target_link_libraries(pokitlms-qwen35-bench PRIVATE PokitLMs::pokitlms)
endif()

if(ANDROID AND POKITLMS_BUILD_ANDROID_APP)
    add_library(pokitlms_android SHARED android/app/src/main/cpp/jni.cpp)
    target_link_libraries(pokitlms_android PRIVATE PokitLMs::pokitlms)
    target_compile_features(pokitlms_android PRIVATE cxx_std_20)
endif()
