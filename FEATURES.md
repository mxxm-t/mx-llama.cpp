# Features

This fork extends upstream llama.cpp with multi-GPU and speculative-decoding
optimizations. Most additions are backend-generic; the gfx906 (VEGA20) kernel
tuning is the only hardware-specific part.

## Multi-stage tensor parallelism (`-tps`)

Upstream's `-sm tensor` runs every layer as one tensor-parallel group across all
GPUs. This fork adds `-tps T` (`--tensor-parallel-size`): split the GPUs into
groups of T, tensor-parallel within each group, and pipeline the layers across
the groups. `T=0` (default) preserves upstream's single-group behavior; `T>0`
requires `n_gpus % T == 0`. Backend-generic.

## Custom GPU AllReduce

An optional peer-write broadcast plus two-shot reduce-scatter / allgather
AllReduce for the tensor-parallel reduction (in addition to upstream's
`allreduce.cu`). F32 on the wire and faster than the RCCL / NCCL ring for token
generation over PCIe. Enable with `GGML_ENABLE_CUSTOM_AR=1`; the fast peer-write
path needs fine-grain PCIe coherence (`HSA_FORCE_FINE_GRAIN_PCIE=1` on any AMD
over PCIe, a no-op on hardware-coherent GPUs and ignored on NVIDIA). Validated
on gfx906.

## MTP speculative-decode optimizations

Opt-in optimizations on top of upstream's `--spec-type draft-mtp` (MTP and the
Qwen3.6 head are upstream), enabled with `LLAMA_ENABLE_MTP_OPT=1`: deferred-prefill
KV staging, a KV-only prefill replay, disabling the draft context's pipeline ring,
and a non-finite-draft fail-safe. Default off uses the standard `draft-mtp` path
with these disabled. Backend-generic.

## Multi-GPU transfer tuning

Hardware-queue handling (`GPU_MAX_HW_QUEUES`) and an optional RCCL point-to-point
stage-transfer path (`GGML_META_XFER_RCCL`) for the multi-stage pipeline.

## gfx906 kernel tuning

Hardware-specific tuning for gfx906 / VEGA20 (MI50, MI60, Radeon VII, Radeon Pro
VII): MMQ tile-width selection, q8_1 quantization, top-k MoE row handling, and
gated-delta-net warp counts.

## Building from source

Requires a ROCm toolchain with gfx906 support (rocBLAS gfx906 kernels, plus RCCL
and rocWMMA for the respective flags). Note gfx906 is deprecated in ROCm 7.x. See
`docs/build.md` for general HIP build background.

```bash
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DLLAMA_OPENSSL=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build --config Release -j
```

## Running

Recommended environment (each variable enables one of the features above):

```bash
export GGML_ENABLE_CUSTOM_AR=1      # custom multi-GPU AllReduce
export HSA_FORCE_FINE_GRAIN_PCIE=1  # peer-write AllReduce fast path (AMD over PCIe, validated gfx906)
export GPU_MAX_HW_QUEUES=8          # MoE throughput on -tps
export LLAMA_ENABLE_MTP_OPT=1       # MTP optimizations (with --spec-type draft-mtp)
```

On a trimmed ROCm runtime (such as the slim Docker image) also set
`HSA_OVERRIDE_GFX_VERSION=9.0.6` so the runtime recognizes the gfx906 GPU. A full
ROCm install detects it automatically and does not need this.

Always pass `--no-mmap -dio` (or `-mmp 0 -dio 1` for `llama-bench`). mmap on the
model file hangs on this stack. Select GPUs with `HIP_VISIBLE_DEVICES` (AMD) or
`CUDA_VISIBLE_DEVICES` (NVIDIA); the example commands below use the AMD form.

```bash
# multi-GPU tensor-parallel server (4 GPUs, full TP)
HIP_VISIBLE_DEVICES=0,1,2,3 llama-server -m model.gguf \
  -ngl 99 -fa 1 -sm tensor -tps 0 --no-mmap -dio --host 0.0.0.0 --port 8080

# 8 GPUs as 4 TP groups of 2 (TP=2, PP=4)
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llama-cli -m model.gguf \
  -ngl 99 -fa 1 -sm tensor -tps 2 --no-mmap -dio

# MTP speculative decode (Qwen3.6 dense)
HIP_VISIBLE_DEVICES=0,1 llama-cli -m Qwen3.6-27B-MTP.gguf \
  -ngl 99 -fa 1 -sm tensor --spec-type draft-mtp --no-mmap -dio
```
