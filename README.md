<!-- fork banner -->
> **MI50-llama.cpp** - single AMD Instinct MI50 (gfx906, 32 GB HBM2) decode and MTP tuning,
> on top of [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp).

### Results

One MI50 32 GB, 300 W / 1825 MHz core / 1125 MHz HBM, ROCm 7.14. Qwen3.8-27B Q8_0, Q8_0 KV cache, `-fa on`, one slot.
Decode is llama-bench TG64 at the stated occupied depth; MTP is llama-server with native MTP, 3 drafts.

| Workload | mx-llama.cpp | MI50-llama.cpp |
|---|---:|---:|
| Decode, 2k context | 21.5 tok/s | 22.8 tok/s |
| Decode, 16k context | 19.6 tok/s | 22.0 tok/s |
| Decode, 64k context | 14.2 tok/s | 20.4 tok/s |
| MTP, 64k reasoning prompt (seed 42, T 1.0) | 21.6 tok/s | 28.8 tok/s |
| MTP, short code / prose prompt (greedy) | - | 51.8 / 43.2 tok/s |
| MTP, 106.7k prompt (greedy) | - | 24.2 tok/s |
| Max context (`-ub 512`) | - | 114,688 |

The repack, concat and recurrent-state changes are output-identical in greedy and seeded checks. GQA6 attention changes FP32 summation order (checked against a 5e-4 gate; 16k perplexity +0.07%), so sampled text can differ from stock.

### Settings

```
llama-server -m Qwen3.8-27B-Q8_0.gguf -ngl 99 -fa on -np 1 \
  -c 66560 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -lm dio \
  --spec-type draft-mtp --spec-draft-n-max 3
```

- Up to ~66k context: `-ub 2048`. Longer: `-ub 512`, up to `-c 114688` (`-ub 2048` above ~90k runs out of FA workspace).
- One slot (`-np 1`) is as fast as several for MTP.

### Changes in this fork

- **GQA6 decode attention** (gfx906, D=256, 24 Q / 4 KV heads, Q8 KV): one K/V read serves all six query heads; 1-5 query rows for MTP verify. Decode keeps 90% of its speed from 2k to 64k.
- **Repack mat-vec tuning**: 8-row workgroups for the fused FFN path, narrow-batch (MTP verify) layout and launch bounds that stop VGPR spills.
- **Conv-state concat**: direct transposed concat, including batches under 32 tokens.
- **Single-slot recurrent ring**: snapshot writes fused into `gated_delta_net` instead of copy + scatter (+7% MTP with one slot).
- **In-place recurrent state**: `gated_delta_net` reads its input state from the cache row, skipping the gather.

### From mx-llama.cpp (gfx906)

- Weight repack for gfx906 (Q8_0 by [iacopPBK](https://github.com/iacopPBK); MXFP4, IQ4_NL, Q4_K, Q5_K, Q6_K, Q5_1)
- Narrow-batch repacked mat-vec for speculative verify, quantized-activation cache, fused up/gate for MoE
- MTP draft heads, snapshot-ring recurrent rollback, DFlash / DSpark
- Multi-stage tensor parallelism, custom peer-write AllReduce

Images and multi-GPU details: **[mxxm/mx-llama.cpp](https://hub.docker.com/r/mxxm/mx-llama.cpp)**.

---
<!-- mx-llama.cpp banner -->

```
-sm tensor          [0 1 2 3 4 5 6 7]                  upstream: one group, all layers
-sm tensor -tps 2   [0 1] -> [2 3] -> [4 5] -> [6 7]   4 groups of 2, pipelined
```

| | |
|---|---|
| **Multi-stage tensor parallelism** | `-tps T` groups the GPUs and pipelines layers across the groups |
| **DeepSeek-V4-Flash on `-sm tensor`** | fork routing kept after upstream's own split landed: GPU-side lightning indexer at any context length, static rollback topology |
| **DeepSeek-V4.1-Flash on `-sm layer`** | architecture support: compressed tiers driven by model metadata, hyper-connection mix shifted by one sublayer, head fold without `output_hc_*` tensors, Engram n-gram hash memory read on demand from two 48.6 GiB tables |
| **Qwen3.8-Flash-Next on `-sm tensor`** | PLE gather table sharded across the TP group (27 GiB on UD-Q4_K_XL, larger at higher quants), opt-in load-time prefault of that table, NextN/MTP draft head, lazy tensor read under `-lm dio` |
| **Speculative decoding** | MTP draft heads on Qwen3.6 and Qwen3.8-Flash-Next, DSpark on DeepSeek-V4-Flash, DFlash, all under tensor parallelism, MTP KV staging, recurrent state rewound from a snapshot ring instead of rebuilt |
| **Custom GPU AllReduce** | peer-write, beats the RCCL ring for generation over PCIe |
| **Weight repack** | GPU-side weight layout for gfx906, on by default, all split modes: Q8_0, MXFP4, IQ4_NL, Q4_K, Q5_K, Q6_K and Q5_1. Q8_0 path by [iacopPBK](https://github.com/iacopPBK) |

[FEATURES.md](FEATURES.md) - flags, measurements, scope. Base: upstream `b10760`.

---
# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
