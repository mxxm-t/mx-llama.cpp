# Features

This fork extends upstream llama.cpp with multi-GPU and speculative-decoding
optimizations. Most additions are backend-generic; the hardware-specific parts
are the gfx906 (VEGA20) kernel tuning and the Q8_0/MXFP4 weight repack, both
MI50 / MI60 / Radeon VII class GPUs.

## Building from source

Requires a ROCm toolchain with gfx906 support (rocBLAS gfx906 kernels, plus RCCL
for `GGML_HIP_RCCL`). Note gfx906 is deprecated in ROCm 7.x. See `docs/build.md`
for general HIP build background.

```bash
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_RCCL=ON \
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

Always pass `-lm dio` (`--load-mode dio`). mmap on the model file hangs on this
stack. The older `--no-mmap` / `-dio` spellings still parse but are deprecated
upstream, and in `llama-bench` they append two separate load modes, so the old
two-flag form runs every benchmark twice. Select GPUs with `HIP_VISIBLE_DEVICES`
(AMD) or `CUDA_VISIBLE_DEVICES` (NVIDIA); the example commands below use the AMD
form.

```bash
# multi-GPU tensor-parallel server (4 GPUs, full TP)
HIP_VISIBLE_DEVICES=0,1,2,3 llama-server -m model.gguf \
  -ngl 99 -fa 1 -sm tensor -tps 0 -lm dio --host 0.0.0.0 --port 8080

# 8 GPUs as 4 TP groups of 2 (TP=2, PP=4)
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llama-cli -m model.gguf \
  -ngl 99 -fa 1 -sm tensor -tps 2 -lm dio

# MTP speculative decode (Qwen3.6 dense)
HIP_VISIBLE_DEVICES=0,1 llama-cli -m Qwen3.6-27B-MTP.gguf \
  -ngl 99 -fa 1 -sm tensor --spec-type draft-mtp -lm dio
```

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
over PCIe, a no-op on hardware-coherent GPUs and ignored on NVIDIA). Decode-size
collectives automatically use two-shot for TP5, TP8, TP10, and TP4 pipeline
stages; standalone TP4 stays on broadcast. Large prompt collectives keep the
RCCL / NCCL size gate. `GGML_TP_AR_TWOSHOT=0` forces broadcast and `=1` forces
two-shot for diagnostics. Validated on gfx906.

## MTP speculative-decode optimizations

Opt-in optimizations on top of upstream's `--spec-type draft-mtp` (MTP and the
Qwen3.6 head are upstream), enabled with `LLAMA_ENABLE_MTP_OPT=1`: deferred-prefill
KV staging, a KV-only prefill replay, disabling the draft context's pipeline ring,
and a non-finite-draft fail-safe. Default off uses the standard `draft-mtp` path
with these disabled. Backend-generic.

## Concurrent lane dispatch

Under `-sm tensor` the meta backend issued each subgraph to its GPUs in device
order, so the AllReduce closing it waited on the last one, and with 80 to 130
such subgraphs per token depending on the model, that stagger was rebuilt at
every one. The lanes are now
issued concurrently, which is bit-exact. The gain tracks how many GPUs share one
tensor-parallel group: measured on Qwen3.6-35B-A3B, +32% token generation on an
8-GPU tensor split and +2.5% on 4, with prefill flat. Under multi-stage `-tps`
it follows the group size rather than the total GPU count. On by default;
`GGML_META_PARALLEL_DISPATCH=0` restores the serial issue. Inert unless at least
two GPUs share a tensor split. CUDA / ROCm sub-backends only.

## Whole-token graph capture

A decode token under `-sm tensor` made one host round trip per AllReduce-bounded
subgraph (80 to 130 per token depending on the model), and the collective billed
the host submission spread at each. Each GPU now records its whole token (subgraph, AllReduce, subgraph, and
so on) into a single CUDA or HIP graph and replays that once per token, which is
bit-exact. Worth +6% token generation on a 4-GPU tensor split, on both a MoE and
a dense model. On 8 GPUs the throughput gain is small but run-to-run spread
drops from 11% to 2.5%. Prefill is unchanged by design. On by default;
`GGML_META_TOKEN_GRAPH=0` restores the per-subgraph dispatch. Requires the
concurrent lane dispatch above. Validated on gfx906.

## DeepSeek-V4-Flash tensor parallelism

Upstream added its own deepseek4 tensor split in b10604, with the same head-split
shape this fork has carried since b10240. The fork keeps its routing: the MLA heads
divide across the tensor-parallel group with the attention-side state mirrored
per lane, the lightning-indexer selection runs on the GPU at any context length
(above 16384 columns it previously fell back to the CPU, which corrupted output
past 65k context), and the indexer top-k needs no cross-lane broadcast because
the fused scores are AllReduce outputs and already bit-identical on every lane.
Byte-deterministic over a 100k-token greedy run, perplexity consistent with
`-sm layer` within 0.3%. Works with multi-stage `-tps`. Validated on gfx906.

deepseek4 also rebuilt its compressed-state and rollback plans on every ubatch, so
the graph changed shape each prefill chunk and the allocation was re-planned every
time. Fixed-width restore and snapshot entries per layout stream make the topology
constant and the allocation is planned once, which is worth far more than it
sounds on long prompts: 8x MI50 `-sm layer` with a 23k prompt, 182 to 759 t/s, and
`-tps 4` generation 23.9 to 26.8 t/s. Three identical requests return identical
output and identical draft acceptance.

## Shared-expert tensor-parallel split

Under `-sm tensor` the DeepSeek shared expert was mirrored: every lane read the
whole ~27 MB/layer Q8_0 shexp each token, equal to the entire routed-expert
read, which is split. The shared expert now routes column-parallel (up/gate)
and row-parallel (down), and the scheduler folds the resulting partial sum into
the ADD that already feeds the per-layer AllReduce, so the reduction adds no
communication and the subgraph count is unchanged. Measured on
DeepSeek-V4-Flash MXFP4 over 8x MI50, 16k prompt: prefill +4-10% in all tensor
configurations, generation +9-15% at `-tps 8` with a draft model and +5% at
`-tps 4` without one; `-tps 4` with a draft pays about 2%. On by default;
`LLAMA_SHEXP_SPLIT=0` restores the mirrored layout. Perplexity shift is
summation-order class (4.0865 vs 4.0800). A DeepSeek-backbone draft model takes
the same routing. Validated on gfx906.

## DSpark drafter under tensor parallelism

The DSpark drafter runs an in-graph argmax over the full vocabulary on logits it
produces, which a vocabulary shard cannot serve, so upstream's
`--spec-type draft-dspark` did not work under `-sm tensor`. The drafter is now
replicated per lane instead of split (a small dense drafter loses more to
per-layer AllReduce than it gains from splitting), the no-vocab sidecar borrows
the target tokenizer, and the target's output projection is replicated so every
device holds the full logit row. Measured on DeepSeek-V4-Flash at `-tps 4` over
8 GPUs: generation 18.0 to 27.3 t/s. The target also stays unrepacked by default:
on the same topology this raised draft acceptance from 63.8% to 83.5% and
generation from 26.8 to 31.5 t/s without changing prefill throughput.
`LLAMA_DSPARK_TARGET_REPACK=1` restores target repacking. Validated on gfx906.

## Allocation layout cache

A change in the scheduler's backend assignment forced a drain-and-reserve before
reallocating, and under `-sm layer` pipeline parallelism that drain hit on every
assignment flip, serializing the pipe. The allocator now caches layouts per
graph topology keyed by the buffer assignment and rebinds without draining when
only the assignment changed, which is bit-exact (outputs byte-identical,
perplexity unchanged). Worth +377% prefill at 100k context on DeepSeek-V4-Flash
across 8 GPUs. On by default; `GGML_GALLOC_LAYOUT_CACHE=0` restores the old
path. Backend-generic.

## Pipeline scheduling

Two scheduler fixes for multi-GPU pipelines, both host-side and bit-exact.

The pinned host buffers that stage graph inputs were allocated at exactly the
requested size. Attention masks grow by one ubatch of columns on every prefill
step, so every step freed and reallocated a slot - and pinned allocation and
free synchronize the device, draining every GPU once per ubatch, with the cost
scaling as the masks widen. They now grow in powers of two.

The ring of input copies bounds how many ubatches can be in flight, and so how
many pipeline stages can compute at once, but its depth was a fixed 4 whatever
the topology: an 8-GPU layer pipeline kept about half its stages busy. The depth
now follows the GPU count. Each slot costs another copy of the graph inputs, so
GGML_SCHED_N_COPIES overrides it either way; tensor-parallel runs keep the depth
their stage count asks for, which is what they want.

Measured on DeepSeek-V4-Flash MXFP4 over 8 MI50 at 100k context, greedy output
byte-identical in every arm:
  -sm tensor -tps 4  prefill 167.0 -> 282.8 t/s  (+69%, staging)
  -sm layer          prefill 426.9 -> 585.4 t/s  (+37%, ring depth)

## Multi-GPU transfer tuning

Hardware-queue handling (`GPU_MAX_HW_QUEUES`) and an optional RCCL point-to-point
stage-transfer path (`GGML_META_XFER_RCCL`) for the multi-stage pipeline.

## gfx906 kernel tuning

Hardware-specific tuning for gfx906 / VEGA20 (MI50, MI60, Radeon VII, Radeon Pro
VII): MMQ tile-width selection, q8_1 quantization, top-k MoE row handling, and
gated-delta-net warp counts.

## BF16 compute on AMD without native bfloat16

On AMD parts predating CDNA and RDNA3, BF16 matmuls compute in F32. rocBLAS has
no tuned bf16 kernel for that hardware, and `compute_type=BF16` also rounds the
F32 activations down to bf16, so F32 is both faster and more faithful. Automatic,
no flag. Worth +18-19% prefill on UD / `*_XL` quants that keep BF16 tensors.
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` selects the old compute type.

## Quantized activation reuse

Several matmuls usually read one activation (q/k/v off a single attn_norm, the
router and gate/up off a single ffn_norm), and each quantized it to q8_1 again.
The quantized copy is now kept and handed to the later matmuls, which is
bit-exact. Worth +2.2-2.6% on prefill and decode. On by default;
`GGML_CUDA_Q8_1_CACHE=0` restores the old behavior. Backend-generic.

## Q8_0 and MXFP4 weight repack (gfx906)

Q8_0 weights upload into a two-plane layout (quants and scales in separate
planes) with tiled MMQ and mat-vec kernels reading it directly, contributed
by DENEB1312. On by default on gfx906, carried by the extra buffer types
like upstream's CPU weight repack, so `--no-repack` disables it; a draft
model always loads canonical weights. Measured on 2x MI50: prefill +12 to
+41% across dense and MoE models and both split modes, generation within a
couple percent of the canonical path. Prefill is bit-exact and perplexity
unchanged; greedy generation can differ within floating-point
reassociation. Model load stages canonical bytes and repacks on the
device, so `-sm layer` loads at vanilla-loader parity and tensor-parallel
loads within about 1.4x of it. Narrow batches, such as the multi-token
steps a speculative verify produces, fuse the MoE up and gate lanes and
size their mat-vec lane group from the tensor shape and the device: a lane
needs enough accumulation steps to cover its reduction, and the grid that
results still has to fill the compute units. That puts them at or ahead of
the canonical path per decode step, worth about 6% on multi-token
prediction with a 35B MoE. Perplexity is unchanged on MoE and moves within
floating-point reassociation on dense (6.7010 to 6.6858 on a 27B dense
model at two tokens), while wide batches stay exact. Validated on gfx906.

MXFP4 weights repack the same way: rows carry the packed nibbles with a
one-byte e8m0 scale plane after them, staying at the canonical 17 bytes per
block so VRAM use does not grow. Measured against the canonical path:
prefill +24% on a 35B MoE (one GPU) and +34% on gpt-oss-120b (two GPUs,
layer split), generation +19% on a 27B dense model, perplexity within
0.02%. Narrow batches and decode share the Q8_0 machinery through
per-type kernel traits: 2-8 token verify batches take one mat-vec per
expert assignment instead of the tiled GEMM (+41% at four tokens on the
35B MoE), decode fuses the up and gate lanes (+6% generation on the 35B
MoE), and tensor-split placement is admitted (+28% prefill on two GPUs
with `-sm tensor`).
