# Android performance measurements

Measurements must use optimized native code. The `benchmark` Android build
is debug-signed and debuggable for ADB model access, with explicit `-O2` native
flags. Select its instrumentation variant with:

```sh
gradle -p android -Ppokitlms.testBuildType=benchmark connectedBenchmarkAndroidTest
```

The real-model test accepts `pokitlms.modelPath`, `pokitlms.maxTokens`, and
`pokitlms.ioThreads`. `PokitLMsAB` logcat events identify each prompt forward,
each selected token, and each decode forward. They include elapsed time,
process user/system CPU time, RSS, process swap, system available RAM, and
system free swap. The summary separates prefill and decode.

The first output token is selected from the final prompt logits. Requesting
one output token measures prefill and selection, with zero decode forwards.
Decode throughput uses `decode_forward_tokens`, rather than the output count.
Progress logging adds a small amount of overhead to the summary durations.

## Initial observations, 2026-10-02

Device: Pixel 9a, Tensor G4, approximately 7.75 GiB usable RAM, Android 16.
Model: Qwen/Qwen3.5-9B, Q4_K_M GGUF, 6,169,341,984 bytes. This is dense;
changing expert I/O workers does not accelerate this model.

- An unoptimized Debug JNI run requesting 8 tokens did not complete within
  600 seconds. Killing the host ADB command did not stop native inference;
  `am force-stop org.pokit.pokitlms` was required to stop it on the device.
- An optimized RelWithDebInfo (`-O2`) run requesting 1 token completed with
  `<think>`. The JNI generation call took 110,911 ms; instrumentation reported
  112.313 seconds including model load and test overhead.
- These runs request different token counts. They do not establish a speedup
  ratio or an optimized decode rate. In particular, the 1-token measurement
  must not be reported as decode tokens/second.
- Optimized benchmark APK compilation was verified by inspecting the native
  compiler flags. Dense Qwen3.5, Qwen3.5 MoE, and Qwen3 MoE tiny-model JNI
  generation passed on this device. Host Dense/MoE tests also verify that
  progress callbacks preserve the output and report all decode steps.

The device's Vulkan property dump identifies Mali-G715, Vulkan 1.4.305,
subgroup size 16, compute subgroup arithmetic, and host-visible device-local
memory. It supports float16/int8 shaders and integer dot products. Initial GPU
work will retain FP32 activations and accumulation and the original GGUF
weight bytes; numerical and token parity must be measured independently of
speed.

## Sources for GPU work

[Strata's native MMVQ](https://github.com/Niko1221/Strata/blob/main/src/kernels/cuda/native_mmvq.cu)
and [prefill MMQ](https://github.com/Niko1221/Strata/blob/main/src/prefill/moe_mmq.cu)
are CUDA/HIP kernels. They are useful references for packed-weight execution
and scheduling. Android Vulkan shader examples come from
[ggml's Vulkan matrix-vector kernel](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp).
The [Vulkan subgroup guide](https://docs.vulkan.org/guide/latest/subgroups.html)
describes capability queries required before using subgroup reductions.
