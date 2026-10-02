cmake_minimum_required(VERSION 3.22)

project(PokitLMs VERSION 0.4.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

option(POKITLMS_BUILD_TESTS "Build PokitLMs tests" ON)
option(POKITLMS_BUILD_BENCHMARK "Build the model-backed inference benchmark CLI" ON)
option(POKITLMS_BUILD_ANDROID_APP "Build the Android JNI bridge" OFF)
option(POKITLMS_USE_VULKAN "Build bounded Vulkan compute-shader linear acceleration" OFF)
option(POKITLMS_GPU_DIAGNOSTICS "Include opt-in Android Vulkan kernel parity checks" OFF)

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

if(POKITLMS_USE_VULKAN)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(POKITLMS_GLSLC glslc HINTS
        "${CMAKE_ANDROID_NDK}/shader-tools/linux-x86_64"
        "${CMAKE_ANDROID_NDK}/shader-tools/darwin-x86_64"
        "${CMAKE_ANDROID_NDK}/shader-tools/windows-x86_64"
        REQUIRED)
    if(ANDROID)
        find_library(POKITLMS_VULKAN_LIBRARY vulkan REQUIRED)
        target_link_libraries(pokitlms PUBLIC ${POKITLMS_VULKAN_LIBRARY})
    else()
        find_package(Vulkan REQUIRED)
        target_link_libraries(pokitlms PUBLIC Vulkan::Vulkan)
    endif()
    set(_shader_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/pokitlms")
    add_custom_command(OUTPUT "${_shader_dir}/vulkan_shaders.hpp"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_shader_dir}"
        COMMAND ${POKITLMS_GLSLC} --target-env=vulkan1.1 -O -DUSE_SUBGROUP=1
            "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/linear.comp" -o "${_shader_dir}/linear-subgroup.spv"
        COMMAND ${POKITLMS_GLSLC} --target-env=vulkan1.1 -O
            "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/linear.comp" -o "${_shader_dir}/linear-workgroup.spv"
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py"
            "${_shader_dir}/vulkan_shaders.hpp"
            linear_subgroup "${_shader_dir}/linear-subgroup.spv"
            linear_workgroup "${_shader_dir}/linear-workgroup.spv"
        DEPENDS src/vulkan/linear.comp tools/embed_spirv.py
        VERBATIM)
    target_sources(pokitlms PRIVATE src/vulkan_linear.cpp "${_shader_dir}/vulkan_shaders.hpp")
    target_include_directories(pokitlms PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(pokitlms PUBLIC POKITLMS_USE_VULKAN=1)
endif()

if(MSVC)
    target_compile_options(pokitlms PRIVATE /W4)
else()
    target_compile_options(pokitlms PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(POKITLMS_BUILD_TESTS)
    enable_testing()
    if(POKITLMS_USE_VULKAN)
        add_executable(pokitlms_vulkan_tests tests/vulkan_linear_test.cpp)
        target_link_libraries(pokitlms_vulkan_tests PRIVATE PokitLMs::pokitlms)
        target_compile_definitions(pokitlms_vulkan_tests PRIVATE
            POKITLMS_QWEN35_SMOKE_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen35-smoke.gguf"
            POKITLMS_QWEN35_MOE_SMOKE_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/qwen35moe-smoke.gguf")
        add_test(NAME pokitlms_vulkan COMMAND pokitlms_vulkan_tests)
    endif()
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
    if(POKITLMS_USE_VULKAN)
        add_executable(pokitlms-vulkan-probe tools/vulkan_probe.cpp)
        target_link_libraries(pokitlms-vulkan-probe PRIVATE PokitLMs::pokitlms)
    endif()
endif()

if(ANDROID AND POKITLMS_BUILD_ANDROID_APP)
    add_library(pokitlms_android SHARED android/app/src/main/cpp/jni.cpp)
    target_link_libraries(pokitlms_android PRIVATE PokitLMs::pokitlms log)
    target_compile_features(pokitlms_android PRIVATE cxx_std_20)
    if(POKITLMS_USE_VULKAN AND POKITLMS_GPU_DIAGNOSTICS)
        target_compile_definitions(pokitlms_android PRIVATE POKITLMS_GPU_DIAGNOSTICS=1)
        target_include_directories(pokitlms_android PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    endif()
endif()
