# PokitLMs

PokitLMs is a native, mobile-first LLM inference backend and Android app in C++20/Kotlin. The runtime is implemented in this repository; it does not embed or fetch llama.cpp. The Android app supports Qwen3-MoE, dense Qwen3.5, and Qwen3.5-MoE GGUF runners.

## Direction

The first target is a focused CPU inference backend for memory-constrained ARM64 devices, with Mixture-of-Experts models as the primary use case. Model weights should be read from storage on demand, routed experts should use a bounded cache, and storage I/O should be measurable separately from compute. Correctness comes first: storage and cache policies must preserve the same weight bytes and model math.

The design takes inspiration from:

- [ds4](https://github.com/antirez/ds4): a deliberately model-focused native inference engine.
- [Strata](https://github.com/Niko1221/Strata): routing-aware expert residency and overlapping CPU/GPU work.
- [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge): bounded expert caching and on-demand weight reads for phones.

These are design references, not dependencies. PokitLMs will implement its own model reader, tensor formats, kernels, and inference loop.

## Backend status

- C++20 tensor and reference operator scaffolding: RMSNorm, linear, feed-forward, and RoPE.
- Portable linear kernels for GGUF Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 weights, with an AArch64 Q4_K path that accumulates packed 4-bit values directly without a decoded float block.
- File-backed expert slice reader with an LRU cache, disk-read accounting, and concurrent positioned reads on POSIX.
- Expert loads request best-effort OS page-cache eviction after copying into the bounded expert cache, avoiding a second cached copy of routed weights.
- Shared random-access model file and GGUF tensor row reader; F32/F16/BF16 rows can be converted without loading a full tensor.
- Disk-backed GGUF matrix-vector dispatch in bounded row batches for F32/F16/BF16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, and Q8_0.
- Automatically sizes matrix row reads around a 256 KiB temporary weight window to reduce small storage reads.
- AArch64 NEON dot-product and weighted-accumulation paths for dense F32, quantized blocks, and grouped-query attention; scalar fallback remains portable.
- Single-token Qwen3-MoE decode path with per-layer GQA KV state, Q/K RMSNorm, RoPE, top-k routing, and cached on-demand expert execution.
- Qwen3.5-MoE support for hybrid Gated DeltaNet/full-attention layers, softmax top-k routing with normalized selected weights, routed experts, and a sigmoid-gated shared expert.
- Qwen3.5-MoE expert weights use a bounded LRU cache sized from available system memory. The next layer's router is also evaluated as an I/O hint to warm predicted expert pages; only the exact current-layer route contributes to inference.
- Tokenizer-backed multi-token generation with greedy or temperature sampling, top-k/top-p filtering, repetition penalty, and EOS stopping.
- Prompt prefill skips vocabulary logits for all but the final prompt token, avoiding repeated output-matrix reads.
- GGUF v3 metadata/tensor-directory reader with known-format payload extent validation; payloads remain file-backed.
- Expert tensor splitting by the GGUF last dimension, ready to feed routed slices into the bounded store.
- Validated GGUF architecture parameters and tensor index for Qwen3-MoE (`qwen3moe`); the token executor is still awaiting comparison with a reference model.
- Qwen3.5 GGUF tensor index and greedy text generation for dense and MoE Qwen3.5 models; prompt prefill skips vocabulary projection until its final token, recurrent convolution/DeltaNet state and bounded full-attention KV state are held and reported separately, and matrix weights stay file-backed.
- `pokitlms-qwen35-bench MODEL.gguf USER_MESSAGE [new_tokens] [kv_window] [kv_precision]` reports prefill/decode throughput, attention and recurrent state memory, and (for MoE) expert-cache capacity and hit counts.
- Tied-output Qwen3-MoE GGUF support: when `output.weight` is absent, the runner reuses `token_embd.weight` for vocabulary projection.
- Qwen3-MoE tensor-name/shape index for the model's base, attention, router, and expert tensors.
- Qwen GPT-2 byte-level BPE encoder/decoder using GGUF vocabulary, merge, token-type, and special-token metadata.
- Streaming MoE routing callback that selects only the top-k expert IDs with O(top-k) routing memory.
- Fixed-capacity ring-buffer KV cache and numerically stable grouped-query causal attention primitive.
- FP32 one-token Gated DeltaNet recurrent-step primitive with reusable scratch and separate key/value head counts, matching Qwen3.5's grouped key heads; stateful depthwise causal convolution, gated RMSNorm, and zero-centered RMSNorm are used by the Qwen3.5 runner.
- FP16 KV residency option; Qwen3-MoE runner uses it by default to halve cache storage while accumulating attention in FP32.
- Optional blockwise Q8_0 KV storage with FP32 attention accumulation for smaller mobile KV footprints; FP16 remains the default. KV resident bytes are exposed to C++/C and printed by the benchmark CLI.
- Bounded sliding KV window: older positions roll out while RoPE positions continue up to the model's advertised context length.
- Expert cache telemetry from the runner and C API: total budget/residency, bytes and operations read, cumulative read time, hits, and misses.
- Expert cache memory is apportioned in whole tensor-slice slots under one aggregate budget, so large expert matrices are not silently left uncached when an equal per-store share cannot hold even one slice.
- Four persistent I/O workers load an expert's gate/up/down slices concurrently and prefetch the next routed expert while the CPU computes the current expert.
- C API version function.
- C ABI opaque Qwen3-MoE handle for creation, serialized text generation, and cache/storage telemetry.
- Qwen3 chat helpers for single messages and multi-turn system/user/assistant history in the C++ and C APIs.
- Multi-turn chat keeps KV state and reuses token-identical history prefixes; generated assistant token IDs are retained so replies are not re-tokenized between turns.
- Dense Qwen3.5, Qwen3.5-MoE, and Qwen3-MoE have deterministic tiny-GGUF generation smoke tests. A real Qwen3.5 0.8B Q4_0 model was loaded and generated on a Pixel 9a (Android 16) through JNI; its first greedy token matched llama.cpp, while later tokens diverged after a close-logit tie. Full-logit parity, real 9B/35B reference comparisons, Qwen3-MoE reference comparison, broader quantized format coverage, and optimized ARM kernels remain incomplete.

## Build

Requirements: CMake 3.22+ and a C++20 compiler.

```bash
cmake -S . -B build -DPOKITLMS_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The tiny GGUF files used for the host and Android generation smoke tests can
be regenerated with:

```bash
python3 tools/generate_qwen35_smoke_model.py tests/data/qwen35-smoke.gguf
python3 tools/generate_qwen35_smoke_model.py --architecture qwen35moe tests/data/qwen35moe-smoke.gguf
python3 tools/generate_qwen35_smoke_model.py --architecture qwen3moe tests/data/qwen3moe-smoke.gguf
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

`pokitlms-bench` runs a real GGUF model and reports model load time, prefill and
decode throughput, total bytes read, and expert-cache latency and hit rate. It
is built by default for host and Android builds:

```bash
./build/pokitlms-bench MODEL.gguf "Explain MoE routing" 64 auto 512 q8 3
```

The positional options are generated token limit, expert-cache MiB (`auto`
uses one quarter of currently available memory, capped at 4 GiB), KV
window size, KV precision (`fp16` or `q8`), and expert I/O workers (`1` to `4`).
Tune I/O workers for the device's storage; the default is `3`. A KV window of
zero selects the mobile default. The output reply
goes to stdout and the measurements go to stderr.

The C++ runner accepts the worker count as its final constructor argument.
The C API exposes the same setting through
`pokitlms_qwen3moe_create_ex_with_io_threads`; existing create functions keep
the default of three workers.

## Android app

Build from the repository root with Gradle 8.7, JDK 17, Android SDK platform
35, NDK 27.2.12479018, and SDK CMake 3.22.1. The app packages `arm64-v8a` and
`x86_64` native libraries; the latter is used by the emulator smoke test.

```bash
gradle -p android assembleDebug
```

The debug APK is written to `android/app/build/outputs/apk/debug/app-debug.apk`.
With an Android 35 x86_64 emulator running, execute the JNI-to-generation
smoke test and Kotlin session tests with:

```bash
gradle -p android connectedDebugAndroidTest testDebugUnitTest
```

The [Android workflow](.github/workflows/android.yml) builds the APK, runs the
local unit tests, and runs this inference smoke test on an emulator for pushes
and pull requests.

The instrumentation suite also has an opt-in real-model JNI check. Set the
`pokitlms.modelPath` instrumentation argument to a GGUF path readable by the
app; it loads the model through a file descriptor and generates one token.
Without that argument, the full-model check is skipped.

The app opens GGUF files through Android's system file picker and keeps the
selected descriptor open while the native runner reads weights from it, so it
does not copy a multi-gigabyte model into app storage. Generation currently
returns a complete reply when decoding finishes; streaming and cancellation
remain future work.

## Near-term backend work

1. Cross-build and compare tokenizer, Qwen3-MoE decode, and each quantized format against reference outputs.
2. Add multi-turn model-specific prompt templates.
3. Expand tensor-format coverage for common GGUF quantizations.
4. Measure and tune the asynchronous storage lanes and cache policy on target ARM64 devices.
5. Measure on target ARM64 devices before choosing cache defaults or adding platform-specific acceleration.
