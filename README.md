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
- Scalar reference linear kernels for GGUF Q4_0 and Q8_0 weights.
- File-backed expert slice reader with an LRU cache, disk-read accounting, and concurrent positioned reads on POSIX.
- GGUF v3 metadata and tensor-directory reader; payloads remain file-backed.
- Streaming MoE routing callback that selects only the top-k expert IDs with O(top-k) routing memory.
- C API version function.
- Tokenizer, quantized kernels, architecture-specific model graph, and end-to-end generation are not implemented yet.

## Build

Requirements: CMake 3.24+ and a C++20 compiler.

```bash
cmake -S . -B build -DPOKITLMS_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Near-term backend work

1. Connect GGUF tensor names and architecture metadata to an explicit supported-model registry.
2. Implement quantized CPU dot products and MoE routing, then compare against small known fixtures.
3. Add tokenizer and model-specific prompt/generation support for the first target architecture.
4. Add async storage reads and compute/I/O overlap after the synchronous path is correct.
5. Measure on target ARM64 devices before choosing cache defaults or adding platform-specific acceleration.
