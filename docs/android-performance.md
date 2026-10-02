# Android performance measurements

Measurements must use optimized native code. The `benchmark` Android build
is debug-signed and debuggable for ADB model access, with explicit `-O2` native
flags. Select its instrumentation variant with:

```sh
gradle -p android -Ppokitlms.testBuildType=benchmark connectedBenchmarkAndroidTest
```

The real-model test accepts the following instrumentation arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `pokitlms.modelPath` | unset | App-readable GGUF path; full-model test is skipped when unset. |
| `pokitlms.maxTokens` | `1` | Maximum generated token count. |
| `pokitlms.ioThreads` | `3` | Expert I/O workers, from 1 to 4. |
| `pokitlms.backend` | `cpu` | `cpu` or `vulkan`; Vulkan accelerates Qwen3.5 linear operations. |
| `pokitlms.gpuTileMiB` | `16` | Capacity of each of the two GPU weight windows. |
| `pokitlms.gpuMode` | `subgroup` | `subgroup` or `workgroup` reduction. |
| `pokitlms.prompt` | `hi` | User message passed to the model's chat template. |
| `pokitlms.expectedReply` | unset | Optional exact reply assertion. |
| `pokitlms.checkVulkan` | unset | Set `true` to run the optional shader diagnostics. |

`PokitLMsAB` logcat events identify each prompt forward,
each selected token, and each decode forward. They include elapsed time,
process user/system CPU time, RSS, process swap, system available RAM, and
system free swap. The summary separates prefill and decode.

The first output token is selected from the final prompt logits. Requesting
one output token measures prefill and selection, with zero decode forwards.
Decode throughput uses `decode_forward_tokens`, rather than the output count.
Progress logging adds a small amount of overhead to the summary durations.

To retain a model already stored in the app's private directory, assemble and
install with `-r` and run instrumentation directly. Uninstalling the target
app removes its private model file. Keep compiler temporary files and logs on
a persistent disk and limit build parallelism:

```sh
export TMPDIR="$HOME/.local/share/pokitlms-tools/tmp"
mkdir -p "$TMPDIR" "$HOME/.local/share/pokitlms-tools/logs"
JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR" \
  gradle -p android --no-daemon --max-workers=2 \
  -Dorg.gradle.jvmargs=-Xmx1500m -Ppokitlms.testBuildType=benchmark \
  :app:assembleBenchmark :app:assembleBenchmarkAndroidTest
adb install -r android/app/build/outputs/apk/benchmark/app-benchmark.apk
adb install -r android/app/build/outputs/apk/androidTest/benchmark/app-benchmark-androidTest.apk
adb shell am instrument -w -r \
  -e class org.pokit.pokitlms.AndroidInferenceSmokeTest#loadsCallerProvidedFullModelWhenConfigured \
  -e pokitlms.modelPath /data/user/0/org.pokit.pokitlms/files/qwen35-9b.gguf \
  -e pokitlms.backend vulkan -e pokitlms.maxTokens 8 \
  -e pokitlms.gpuTileMiB 16 -e pokitlms.gpuMode subgroup \
  -e pokitlms.prompt hi \
  org.pokit.pokitlms.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s PokitLMsAB:I > "$HOME/.local/share/pokitlms-tools/logs/android-vulkan.log"
```

The example assumes the model is already present at the specified path.
To run only kernel diagnostics, select
`AndroidInferenceSmokeTest#verifiesVulkanKernelsWhenConfigured` and pass
`-e pokitlms.checkVulkan true`. These checks are compiled into the benchmark
variant through `POKITLMS_GPU_DIAGNOSTICS=ON`.

If an ADB command times out or disconnects, native inference may continue on
the phone. On reconnection, use
`adb shell am force-stop org.pokit.pokitlms` to stop that run before starting
another measurement.

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
memory. It supports float16/int8 shaders and integer dot products.

## Native Vulkan implementation

The optional Vulkan backend supports F32, F16, BF16, Q4_0, Q4_1, Q8_0,
Q4_K, Q5_K, and Q6_K matrix weights. Other formats continue through the CPU
kernels. It uses the original packed GGUF bytes with FP32 activations and
accumulation, without activation requantization or dropping routed experts.
The runtime is implemented in PokitLMs and has no llama.cpp dependency.

Two mapped weight windows, each 16 MiB by default, bound GPU weight
residency. Positioned reads fill the next window while the current window
executes on the GPU. Fence completion is required before reusing a window;
inputs and outputs have separate buffers. Subgroup and workgroup reductions
are available. Floating-point reduction order differs from CPU arithmetic,
so this is numerical parity rather than a bitwise guarantee.

The host CMake option is `POKITLMS_USE_VULKAN=ON`, with Python 3, Vulkan
headers/loader, and `glslc` required. Android Gradle builds enable Vulkan and
find the NDK's host shader compiler. The C++ API installs a shared
`VulkanLinearBackend` through `Qwen35Runner::set_linear_backend`; JNI accepts
`useVulkan`, `gpuTileMiB`, and `gpuSubgroups` when loading the model. Explicit
Vulkan initialization errors are surfaced to the caller. Qwen3-MoE remains
on the CPU.

## Initial Vulkan measurements, 2026-10-02

These measurements describe the initial implementation, before subsequent
cache-maintenance and full-subgroup handling fixes. They are single runs,
with no controlled whole-model page-cache state.

All three completed runs used the same 9B model, `hi`, greedy generation,
9 prompt tokens, 8 generated tokens, FP16 KV storage, and 7 decode forwards.
Vulkan used 16 MiB windows and subgroup reduction.

| Device/backend | Prefill ms | Decode ms | Decode forwards/s | Generation total s |
| --- | ---: | ---: | ---: | ---: |
| Host CPU | 103,986.0 | 92,179.4 | 0.07594 | 196.165 |
| Host GTX1080 Vulkan | 77,753.2 | 82,902.3 | 0.08444 | 160.656 |
| Pixel9a Mali-G715 Vulkan | 95,681.990 | 82,157.302 | 0.08520 | 177.842 |

The host Vulkan timings are lower than the contemporary host CPU timings,
but uncontrolled cache state prevents treating the ratio as a general
speedup. The phone Vulkan run in this table used the earlier shader and
runtime before the full-subgroup and cache-maintenance fixes.

The selected token IDs matched across all three completed runs:

```text
248068,271,248069,271,9419,0,2500,628
```

They decode to `<think>\n\n</think>\n\nHello! How can`. This checks only one
eight-token continuation; broader response accuracy and full-model logits
remain unverified. No full 35B inference was run for these Vulkan measurements.

A later CPU run on the same Pixel 9a and model used the rebuilt benchmark APK,
the same `hi` prompt, and the same eight-token limit. It completed in
174,997 ms, with 90,462.123 ms prefill and 84,529.856 ms for seven decode
forwards. Its eight token IDs and displayed reply matched the Vulkan run
above. Since the comparison Vulkan run used the earlier APK, this is not a
controlled before/after comparison; the rebuilt Vulkan run remains pending.

### Weight delivery and cache state

The host Vulkan run recorded 78,500,200,448 logical weight bytes, 153,963 ms
inside weight reads, and 3,344.55 ms of timestamped GPU computation. The
phone recorded the same logical weight bytes, 125,029.589 ms inside reads,
57,295.022 ms of GPU computation, and 31,791.122 ms waiting on fences.

These counters overlap: reading the next tile can run during GPU computation,
and fence waits can overlap the timestamped GPU interval. They must not be
added together to reconstruct wall time. GPU timestamps also exclude host
read, command preparation, and cache maintenance. `read_ms` includes copying
from the filesystem/page cache into mapped Vulkan memory, not disk latency
alone. `weight_bytes` and `model_bytes_read` count logical reads, not physical
storage traffic.

An earlier host CPU measurement was approximately 0.555 decode forwards/s,
but its cache state was not recorded. A later `mincore` snapshot found only
39.76% of the model resident; this suggests cache/storage effects could
confound comparisons, but does not establish residency during prior runs.

A bounded probe repeatedly read the same warmed 4,718,592-byte matrix window
eight times, totaling 37,748,736 bytes. Ordinary CPU reads took 1.92845 ms;
reads into the Vulkan mapping took 3.65394 ms. Both recorded zero physical
read bytes through `/proc/self/io`. GPU compute took 1.5064 ms and the full
GPU probe took 8.09842 ms. This demonstrates a mapped-write cost in that
small cached case, while showing that the full run's hundreds-of-MB/s read
rate cannot be explained by those warm mapped writes alone. The probe does
not reproduce full-model cache churn or establish a single cause.

During the completed phone CPU run, the process recorded approximately
77.0 GB of physical storage reads across 78.5 GB of logical weight reads.
Each prefill step added around 4.4 GB of physical reads, with RSS near 250 MiB
and system `MemAvailable` around 1.2 GiB. This points to repeated storage
reads as a major cost for this 9B phone run. The process memory stayed around
250 MiB while the test ran; further clock and thermal measurements are needed
for a controlled GPU comparison.

On NVIDIA, the initial memory-selection policy favors uncached host-visible
device-local BAR memory over cached system memory. On Mali, it favors cached
host-visible device-local memory. Vulkan documents BAR memory as appropriate
for small uploads and describes lower peak write performance than system
memory for large transfers. Alternative memory choices require an A/B
comparison of read, GPU, and total time. [Vulkan memory allocation guidance](https://docs.vulkan.org/spec/latest/chapters/memory.html)

### Numerical checks

Randomized packed-kernel comparisons, multiple stream-window reuse, both
reduction modes, and tiny dense/MoE generation checks passed on the GTX1080
and Mali-G715. These exercises support kernel correctness under their tested
shapes and values, rather than full-model accuracy.

The real 9B tensor probe compared eight-row slices from the start, middle,
and end of 282 matrices: 846 slice comparisons, using a deterministic input.
Maximum absolute GPU/CPU error was:

| Format | Maximum absolute error |
| --- | ---: |
| F32 | 3.45707e-6 |
| Q8_0 | 7.15256e-7 |
| Q4_K | 1.07288e-6 |
| Q5_K | 7.7486e-7 |
| Q6_K | 9.53674e-7 |

The probe tolerance is `1e-4 + 5e-5 * abs(cpu_value)`, with nonfinite GPU
outputs rejected. It checks bounded matrix slices, not every output row or
the complete recurrent/attention inference path. F16, BF16, Q4_0, and Q4_1
were covered by synthetic kernel tests rather than this real 9B format set.

`pokitlms-vulkan-probe MODEL.gguf [timing_tensor] [weights=device|cached]` reproduces the bounded
slice and warm-read checks without constructing a model runner or starting
generation. The default timing tensor is `blk.0.ffn_gate.weight`; the timing
window is capped at 8 MiB. The benchmark reports all selected token IDs so
full-generation CPU/GPU comparisons can be made separately.

After limiting mapped cache maintenance to the used byte ranges and requiring
full subgroups where supported, the host checks still passed. A repeated warm
memory-policy comparison selected NVIDIA weight memory flags `7` for mapped
device memory and `14` for cached system memory. The eight-read device-memory
probe took 5.38845 ms total (read 3.52492 ms, GPU 1.53078 ms), versus 9.25009 ms
for cached system memory (read 3.17701 ms, GPU 5.6193 ms). Both recorded zero
physical reads. The device-memory default was retained; choosing faster CPU
copy memory alone did not improve this probe's total time.

The 35B file also passed a bounded matrix check: 514 matrices and 1,542 slices,
with maximum absolute errors below `3e-6` across F32, Q8_0, Q4_K, Q5_K, Q6_K,
and BF16. This only reads small matrix regions and does not run 35B generation.

## Sources for GPU work

[Strata's native MMVQ](https://github.com/Niko1221/Strata/blob/main/src/kernels/cuda/native_mmvq.cu)
and [prefill MMQ](https://github.com/Niko1221/Strata/blob/main/src/prefill/moe_mmq.cu)
are CUDA/HIP kernels. They are useful references for packed-weight execution
and scheduling. Android Vulkan shader examples come from
[ggml's Vulkan matrix-vector kernel](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp).
The [Vulkan subgroup guide](https://docs.vulkan.org/guide/latest/subgroups.html)
describes capability queries required before using subgroup reductions.
