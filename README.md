# PokitLMs

PokitLMs is a native, mobile-first LLM inference backend in C++20. The runtime is being built in this repository; it does not embed or fetch llama.cpp. There is currently no application frontend.

## Direction

The first target is a focused CPU inference backend for memory-constrained ARM64 devices, with Mixture-of-Experts models as the primary use case. Model weights should be read from storage on demand, routed experts should use a bounded cache, and storage I/O should be measurable separately from compute. Correctness comes first: storage and cache policies must preserve the same weight bytes and model math.

The design takes inspiration from:

- [ds4](https://github.com/antirez/ds4): a deliberately model-focused native inference engine.
- [Strata](https://github.com/Niko1221/Strata): routing-aware expert residency and overlapping CPU/GPU work.
- [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge): bounded expert caching and on-demand weight reads for phones.

These are design references, not dependencies. PokitLMs will implement its own model reader, tensor formats, kernels, and inference loop.

## Backend status

- C++20 tensor and reference operator scaffolding: RMSNorm, linear, feed-forward, and RoPE.
- Scalar reference linear kernels for GGUF Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, and Q8_0 weights.
- File-backed expert slice reader with an LRU cache, disk-read accounting, and concurrent positioned reads on POSIX.
- Expert loads request best-effort OS page-cache eviction after copying into the bounded expert cache, avoiding a second cached copy of routed weights.
- Shared random-access model file and GGUF tensor row reader; F32/F16/BF16 rows can be converted without loading a full tensor.
- Disk-backed GGUF matrix-vector dispatch in bounded row batches for F32/F16/BF16, Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, and Q8_0.
- Automatically sizes matrix row reads around a 256 KiB temporary weight window to reduce small storage reads.
- AArch64 NEON dot-product and weighted-accumulation paths for dense F32, quantized blocks, and grouped-query attention; scalar fallback remains portable.
- Single-token Qwen3-MoE decode path with per-layer GQA KV state, Q/K RMSNorm, RoPE, top-k routing, and cached on-demand expert execution.
- Tokenizer-backed multi-token generation with greedy or temperature sampling, top-k/top-p filtering, repetition penalty, and EOS stopping.
- Prompt prefill skips vocabulary logits for all but the final prompt token, avoiding repeated output-matrix reads.
- GGUF v3 metadata/tensor-directory reader with known-format payload extent validation; payloads remain file-backed.
- Expert tensor splitting by the GGUF last dimension, ready to feed routed slices into the bounded store.
- Validated GGUF architecture parameters and tensor index for Qwen3-MoE (`qwen3moe`); the token executor is still awaiting comparison with a reference model.
- Qwen3-MoE tensor-name/shape index for the model's base, attention, router, and expert tensors.
- Qwen GPT-2 byte-level BPE encoder/decoder using GGUF vocabulary, merge, token-type, and special-token metadata.
- Streaming MoE routing callback that selects only the top-k expert IDs with O(top-k) routing memory.
- Fixed-capacity ring-buffer KV cache and numerically stable grouped-query causal attention primitive.
- FP16 KV residency option; Qwen3-MoE runner uses it by default to halve cache storage while accumulating attention in FP32.
- Bounded sliding KV window: older positions roll out while RoPE positions continue up to the model's advertised context length.
- Expert cache telemetry from the runner and C API: total budget/residency, bytes and operations read, cumulative read time, hits, and misses.
- Expert cache memory is apportioned in whole tensor-slice slots under one aggregate budget, so large expert matrices are not silently left uncached when an equal per-store share cannot hold even one slice.
- One persistent storage worker prefetches the next routed expert while the CPU computes the current expert.
- C API version function.
- C ABI opaque Qwen3-MoE handle for creation, serialized text generation, and cache/storage telemetry.
- Qwen3 single-message chat helper in the C++ and C APIs, including the standard user/assistant boundary tokens.
- Model-specific chat-template handling, broader quantized format coverage, and optimized ARM kernels are not implemented yet. The decode path is currently compile-verified but has not been compared against a reference model output.

## Build

Requirements: CMake 3.24+ and a C++20 compiler.

```bash
cmake -S . -B build -DPOKITLMS_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Build the native ARM64 Android backend with an installed NDK:

```bash
cmake -S . -B build-android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DPOKITLMS_BUILD_TESTS=OFF
cmake --build build-android-arm64 -j
```

## Near-term backend work

1. Cross-build and compare tokenizer, Qwen3-MoE decode, and each quantized format against reference outputs.
2. Add model-specific prompt templates and tied-output model support.
3. Expand tensor-format coverage for common GGUF quantizations.
4. Add async storage reads and compute/I/O overlap after the synchronous path is validated.
5. Measure on target ARM64 devices before choosing cache defaults or adding platform-specific acceleration.
