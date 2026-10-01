# PokitLMs

PokitLMs is a mobile-first LLM inference runtime focused on low latency, low memory usage, and efficient execution on ARM64 devices.

## Goals

- Fast token generation on smartphones
- C++20 core with a small C ABI
- ARM64/NEON first, with portable scalar fallbacks
- Quantization-friendly tensor and kernel design
- Android and iOS integration without binding the core runtime to either platform
- Minimal dependencies

## Initial architecture

```text
apps / bindings
      |
    C API
      |
 runtime
      |
 transformer ops
      |
 tensor + kernels
      |
 scalar / ARM NEON / future Metal & Vulkan
```

## Build

Requirements:

- CMake 3.24+
- A C++20 compiler

```bash
cmake -S . -B build -DPOKITLMS_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Roadmap

1. Tensor/storage primitives and reference kernels
2. RMSNorm, Linear, RoPE, attention and KV cache
3. Minimal decoder-only transformer
4. Model loader and tokenizer
5. ARM64 NEON kernels
6. INT8 and INT4 weight-only quantization
7. Android JNI and iOS bindings
8. Metal/Vulkan backends

## Status

Early-stage runtime scaffolding. The first milestone is a correct CPU reference implementation before architecture-specific optimization.
