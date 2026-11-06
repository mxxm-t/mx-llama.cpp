# Features

This fork extends upstream llama.cpp with multi-GPU and speculative-decoding
optimizations. Most additions are backend-generic; the hardware-specific parts
are the gfx906 (VEGA20) kernel tuning and the weight repack, both
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
generation over PCIe. On by default, `GGML_ENABLE_CUSTOM_AR=0` turns it off; the fast peer-write
path needs fine-grain PCIe coherence (`HSA_FORCE_FINE_GRAIN_PCIE=1` on any AMD
over PCIe, a no-op on hardware-coherent GPUs and ignored on NVIDIA). Decode-size
collectives automatically use two-shot for TP5, TP8, TP10, and TP4 pipeline
stages. A standalone TP4 group crosses to two-shot at 8192 elements, which keeps the
one-token message of ordinary decode on broadcast and moves the wider message a
speculative verify batch sends, where broadcast's per-peer write is 74% of the call.
Large prompt collectives keep the RCCL / NCCL size gate, so prefill is unaffected.
Qwen on four MI50, 15k prompt, 512 generated, identical output and draft acceptance:

```
Qwen3.8-27B-Q8 + DFlash2 drafter   -tps 4   43.9 -> 48.0 t/s decode
Qwen3.6-27B-Q8 + MTP head          -tps 4   66.7 -> 72.2 t/s decode
Qwen3.8-27B-Q8 no drafter          -tps 4   47.4 -> 47.2 t/s decode  (guardrail)
```

`GGML_TP_AR_TWOSHOT=0` forces broadcast and `=1` forces two-shot for diagnostics,
and `GGML_TP_AR_TWOSHOT_MIN_NE` sets the crossover in elements. Validated on gfx906.

## MTP speculative-decode optimizations

Opt-in optimizations on top of upstream's `--spec-type draft-mtp` (MTP and the
Qwen3.6 head are upstream), enabled with `LLAMA_ENABLE_MTP_OPT=1`: deferred-prefill
KV staging, a KV-only prefill replay, disabling the draft context's pipeline ring,
and a non-finite-draft fail-safe. Default off uses the standard `draft-mtp` path
with these disabled. Backend-generic.

## Recurrent state rollback (snapshot ring)

Recurrent and hybrid models checkpointed their state by whole planes, so a rejected
speculative draft had no cheap way back and the state was rebuilt rather than rewound.
That is what made MTP drafting on a delta-net model cost more than it saved. Each
sequence now keeps its snapshots in a ring of physical planes with a head and a valid
depth, the delta-net kernels scatter per-token snapshot rows as they run, and a graph
that fails invalidates the affected sequences instead of leaving a half-written plane
visible. Measured on 4x MI50 with Qwen3.8-Flash-Next MTP UD-Q4_K_XL, 554-token prompt
at draft depth 2: speculative generation 29.8 to 42.4 t/s on `-sm layer` and 16.4 to
42.5 t/s on `-sm tensor`, where before the change speculating was slower than not
speculating at all. Plain generation goes 30.3 to 31.4 t/s with byte-identical output,
and Qwen3.6-35B-A3B-Q4_K_M single-GPU prefill is unchanged, 955 to 953 (guardrail). No
flag: rollback engages when a caller asks for it with a single sequence, and it is
clamped off above one sequence and below a minimum micro-batch with a warning. It lives
in the recurrent memory layer, so every delta-net model shares it. Backend-generic, the
delta-net kernels are validated on gfx906.

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
drops from 11% to 2.5%. Prefill is unchanged by design.

A pipeline records one graph per stage and replays the stage transfer between them,
so a multi-stage split is captured as well, which it was not before:

```
DeepSeek-V4.1-Flash MXFP4  -tps 2, 10 GPUs, 5 stages   23.0 -> 23.9 t/s decode   (fork's pre-v0.9.0 V4.1 implementation)
gpt-oss-120b MXFP4         -tps 2,  8 GPUs, 4 stages   82.2 -> 89.3 t/s decode
Qwen3.8-27B-Q8             -tps 2,  4 GPUs, 2 stages   29.6 -> 31.3 t/s decode
```

While the token graph owns the token, each lane's own per-subgraph graph cache is
switched off, because it records the same kernels underneath and reuses each capture
about twice before the next graph displaces it. That is worth 43.5 -> 47.1 t/s decode
on a dense 27B at four lanes and nothing on a 30-layer MoE, so the gain is model
dependent while the cost never is. On by default;
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

## DeepSeek-V4.1-Flash architecture support

DeepSeek-V4.1-Flash support is upstream llama.cpp's (#31086, v0.8.0).
The fork carried its own implementation before that, under the same `deepseek41` architecture name but with a different GGUF layout, and the v0.9.0 merge replaced it with upstream's.
GGUFs converted for the fork's implementation carry `deepseek41.engram.layer_ids` and `blk.N.engram_embed` tensors; they stop at load with a missing-key error and must be reconverted with `convert_hf_to_gguf.py`.
Upstream's format keeps all Engram tables in one lazily read `per_layer_token_embd` tensor (BF16 from the converter; `llama-quantize` can requantize it) and gathers the rows in the graph.
`LLAMA_PLE_SHARD` applies to Qwen3.8-Flash-Next only, not to these tables.

The fork keeps its tensor-parallel routing for V4.1: the design-B attention head split and the shared-expert split.
Speculative decoding uses upstream's V4.1 DSpark drafter, selected by `hyper_connection.single_pass` in the sidecar.
`LLAMA_DSV41_SPARSE=0` disables the sparse attention path.
The fork's diagnostic gates `LLAMA_DSV41_NO_ENGRAM`, `LLAMA_DSV41_NO_COMPRESS`, `LLAMA_DSV41_QNORM` and `LLAMA_DSV41_HOST_INPUT_THREADS` are gone with the old implementation.
V4.1 measurements elsewhere in this file were taken on the old implementation and have not been repeated on upstream's.

## Qwen3.8-Flash-Next tensor parallelism

Qwen3.8-Flash-Next carries a PLE n-gram table - 27465 MiB on the UD-Q4_K_XL quant, larger at
higher quants - that is otherwise host
resident and demand paged, so every token pays a PCIe read to gather its rows and
prefill is bound by that gather rather than by compute. `LLAMA_PLE_SHARD=1` splits
the table by whole hash heads under `-sm tensor`, one segment owned per device, and
get_rows zeroes the rows a device does not own so the partial vectors reduce through
the existing AllReduce. On 4x MI50 with the UD-Q4_K_XL quant: 4096-token prefill 224
to 500 t/s, 512-token generation 18 to 37 t/s, wiki perplexity 2.3108 to 2.3052
(within error). The host staging copy is still retained after upload, so peak host
memory is unchanged - this buys throughput, not footprint.

The 103 GiB model does not fit alongside a 27 GiB gather table in host memory, so
the loader maps lazily-read tensors even under `-lm dio`: the mapping is virtual,
prefetch is zero and the range is never populated, which avoids whole-model mmap's
page thrashing while still letting the table be demand paged.

The table can instead be warmed at load: `LLAMA_PLE_PREFAULT=1` touches one byte per
page of it from eight threads once the weights are in, so the first request does not
fault the rows in one at a time. That moves the read out of the first request and into
load time, so how much it is worth is bounded by storage throughput, and it buys
nothing once the table is already in page cache - it pays on a fresh process, not on a
warm one. The log line reports the size touched and how long it took, so the cost is
visible per machine. Off by default, inert when the table is not host resident, and it
uses no VRAM.

A NextN/MTP draft head is supported with `--spec-type draft-mtp`, converted by
`convert_hf_to_gguf.py --mtp`. Draft acceptance runs 75-90 percent at `n_max 2` and
is strongly text dependent (46 to 90 percent across prompts). With the recurrent
snapshot ring above it is a throughput win in both split modes on this quant - see
that section for the numbers - where previously the multi-GPU verify under `-sm tensor`
cost more than the drafting saved. Acceptance still tracks the text, so measure on your
own prompts.

Speculating used to cost most of the prefill throughput on this architecture, because two
graph shapes moved from chunk to chunk and the scheduler could not keep a plan across them.
Changing the NextN mode altered the graph output requirements without invalidating the
reservation, so the following graph ran on a plan sized for the previous mode, and the
rollback convolution snapshot graph emitted one window per snapshot the batch happened to
carry, so its node count tracked history length. The snapshot graph now emits one window
per ring plane and crops back to the snapshots actually present, which holds the node count
constant while producing the same rows. On 4x MI50 with the MTP UD-Q4_K_XL quant, a
15.8k-token prompt under `-sm layer` goes 421 to 1047 t/s of prefill, against 1184 t/s for
the same prompt without speculation - so drafting now costs about a tenth of prefill rather
than two thirds. Generation and draft acceptance are unchanged and the output is
byte-identical. On by default, with no flag.

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

## DFlash drafters under tensor parallelism

A DFlash or DFlash2 drafter without its own output tensor reads the target's lm head,
which `-sm tensor` splits across the group, so the drafter's selector runs top-k over
logits that each lane holds only a slice of. The meta backend now gathers a source that
is split along axis 0 before an op that reads whole rows, for every such op (top-k,
argsort, sum-rows, cumsum, mean, argmax, the norms and the cross-entropy pair), stages
each lane's slice into one tensor and repoints the consumers at it, so the drafter sees
the whole vocabulary. Without it, the drafter aborted at partition time instead of
serving. Qwen3.8-27B-Q8 with its DFlash2 drafter on two MI50, `--spec-draft-n-max 5`,
against the same model under `-sm layer`:

```
620-token prompt, 128 generated     320 -> 457 t/s prefill   43.8 -> 52.1 t/s decode   100 of 128 accepted in both
16k summary, 512 generated          485 -> 544 t/s prefill   29.5 -> 39.4 t/s decode   349/810 vs 351/797 accepted
```

A graph that plans no gather is untouched, which a plain Qwen and DeepSeek-V4.1 confirm
bit-for-bit. `-tpsd` runs the drafter on fewer lanes than the target. Validated on gfx906.

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

## Incremental stage transfer for persistent caches

A multi-stage `-sm tensor` pipeline hands every mirrored tensor a later stage reads across the stage boundary.
When the reader is a view of a persistent cache, the boundary tensor used to be the writer's set_rows output, which is the whole cache view, so each hop copied the full cache on every token, and each reading layer's own view of a shared cache added one more full copy.
DeepSeek-V4.1 layers after a source layer read the source layer's caches, and on ten MI50 at `-tps 2` (measured on the fork's pre-v0.9.0 V4.1 implementation) the stage 3 hop moved about 244 MB per token, which is where tensor decode lost to layer mode and why the prefill pipeline never filled.
The meta backend now keeps a host copy of the small integer inputs it is handed (the KV cell indices) and transfers a set_rows boundary as the dense cell range those indices name, a row block of each lane's own cache, exact bytes.
A pure view of a persistent cache is served by the cache's writer once instead of one full copy per reading layer.
Sparse or unknown indices fall back to the full copy, and `GGML_META_XFER_ROWS=0` restores the previous transfers.
Measured on ten MI50 with a 16965-token prompt at temperature 0, Engram off, `-sm tensor -tps 2`, with the greedy top-5 records bit-identical across three requests and two boots:

```
prefill  214 -> 601 t/s     decode  7.2 -> 21.5 t/s     (-sm layer on the same prompt: 600 to 650 and 14.1)
```

Only multi-stage tensor splits go through this path, layer mode and single-stage splits do not run it, and a model whose layers read only their own caches moves nothing new.

## Stage-local mirrored weights

A multi-stage `-sm tensor` split used to place a full copy of every mirrored weight (norms, routers, unsplit projections) on every card, including cards of stages that never read it.
llama now names the lanes that read a mirrored layer weight, which are the lanes of its own stage plus the neighbouring stage for the first and last layer of a stage, because the stage seam falls between the last AllReduce of one layer and the first split weight of the next.
The meta backend allocates, writes and reads the copy only on those lanes, deciding presence through the view root.
A node with no owning stage of its own follows the weight instead of the walk, taking the lowest stage whose lanes hold every stage-local mirrored weight it reads, so a draft context narrower than its target serves rather than aborting at load.
A partition-time check still aborts when two weights a node needs sit on different stages, which no placement can satisfy.
Compute placement does not change, and `LLAMA_STAGE_LOCAL_MIRROR=0` restores a copy on every lane.
Largest card VRAM, on ten MI50 unless noted, greedy outputs with top-5 records bit-identical with the copies on and off, prefill and decode unchanged:

```
DeepSeek-V4.1-Flash MXFP4   -tps 2, 32k, ub 512, repack on   out of memory at load -> serves with 1.2 GiB free per card   (pre-v0.9.0 V4.1)
DeepSeek-V4.1-Flash MXFP4   -tps 2, 32k, ub 512, repack off  31374 -> 30496 MiB   (pre-v0.9.0 V4.1)
DeepSeek-V4-Flash MXFP4     -tps 2                            18876 -> 17936 MiB
DeepSeek-V4-Flash MXFP4     -tps 4, 8 GPUs                    21780 -> 21178 MiB
Qwen3.8-Flash-Next Q8_0     -tps 2                            17230 -> 16504 MiB
MiniMax-M2.1 Q8_0           -tps 2                            26501 -> 26357 MiB
gpt-oss-120b MXFP4          -tps 2, 4 GPUs                    16297 -> 16273 MiB  (guardrail)
```

The saving is the mirrored mass, so models whose large matrices are all split (gpt-oss, Gemma 4, dense Qwen) change by a few MiB, and layer mode and single-stage splits are untouched.

## Tied output heads under tensor parallelism

A model whose output head is tied to its token embedding (the gemma-4 family among
others) loads the head as a duplicate of `token_embd.weight` and the duplicate keeps
that name. The tensor split rules go by name, so the head used to fall to the mirrored
default together with the embedding table. Every lane then ran the whole head each
token. The duplicate is only ever read by a matrix multiply, so it now takes the same
vocabulary split as an untied `output.weight`. gemma-4-26B-A4B-Q8 on four MI50 at
`-tps 4`: generation 104 to 117 t/s, prefill unchanged, 561 MiB less per card, output
identical. Layer and single-device runs never mirrored the head and are unchanged.

## gfx906 kernel tuning

Hardware-specific tuning for gfx906 / VEGA20 (MI50, MI60, Radeon VII, Radeon Pro
VII): MMQ tile-width selection, q8_1 quantization, top-k MoE row handling, and
gated-delta-net warp counts.

## BF16 compute on AMD without native bfloat16

On AMD parts predating CDNA and RDNA3, BF16 matmuls over more than 32 columns compute in F32. rocBLAS has
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

## Q8_0, MXFP4, K-quant and Q5_1 weight repack (gfx906)

Weights of the types below upload into a repacked layout (quants and scales
in separate planes, rows de-aliased) that the gfx906 MMQ and mat-vec kernels
read directly, so prefill stops paying for per-block scale gathers. The Q8_0
path was contributed by DENEB1312; MXFP4, the K-quant types and the legacy
Q5_1 follow it through per-type kernel traits. On by default on gfx906,
carried by the extra buffer types like upstream's CPU weight repack, so
`--no-repack` disables it
(`--repack 0` or `-nr 1` in llama-bench); a draft model always loads canonical weights. Model
load stages canonical bytes and repacks on the device, so `-sm layer` loads
at vanilla-loader parity and tensor-parallel loads within about 1.4x of it.
Every type is admitted under `-sm tensor` and multi-stage `-tps`, where each
lane slice repacks. VRAM use stays at the canonical size for every type.

| type | repacked layout | prefill vs canonical | generation vs canonical | numerics |
|---|---|---|---|---|
| Q8_0 | two planes, int8 quants and f16 scales | +12 to +41% (dense and MoE, 2x MI50, both split modes) | within a couple percent | prefill bit-exact, PPL unchanged |
| MXFP4 | packed nibbles, one-byte e8m0 scale plane | +24% (35B MoE, 1 GPU), +34% (gpt-oss-120b, 2 GPU layer), +28% (2 GPU tensor) | +19% (27B dense) | PPL within 0.02% |
| IQ4_NL | nibble plane, f16 scale plane, k-value table | +133% (1 GPU), +112% (2 GPU tensor) | +5% (1 GPU), -2% (2 GPU tensor) | byte-identical boots, PPL 7.4748 vs 7.4741 |
| Q6_K | de-aliased lows, highs plane, per-16 scale pairs, f16 d plane | +69% (1 GPU), +59% (2 GPU tensor) | +6% (1 GPU), -2% (2 GPU tensor) | byte-identical boots, PPL 7.3822 vs 7.3823 |
| Q5_K_M | Q4_K planes plus a fifth-bit word per sub-block | +55% (1 GPU), +50% (2 GPU tensor) | +1% (1 GPU), -5% (2 GPU tensor) | byte-identical boots, PPL 7.4895 vs 7.4929 |
| Q4_K_M | de-aliased nibbles, scale/min record, half2 d and dmin | +2% (1 GPU), +5% (2 GPU tensor) | +7% (1 GPU), -2% (2 GPU tensor) | byte-identical boots, PPL 7.4319 vs 7.4267 |
| Q5_1 | nibble plane, fifth-bit word, half2 d and m | +84% (1 GPU), +74% (2 GPU tensor) | flat (1 GPU and 2 GPU tensor) | byte-identical output on 1 GPU, PPL 6.6031 vs 6.6351 |

The K-quant rows are Qwen3-14B on MI50, pp512, tg128, one card in layer
mode and two cards with `-sm tensor -tps 2`; Q4_K and Q5_K carry the affine
scale and min pair and fold the activation sum through the q8_1 block sums.
The Q5_1 row is a Qwen3.6-27B requantized to Q5_1, as no released Q5_1 build
of it exists; on a Qwen3.8-Flash-Next MoE carrying 43 Q5_1 tensors the same
repack is +11% prefill on four cards with `-sm tensor`.
Greedy generation can differ from the canonical kernels within
floating-point reassociation on every type.

Narrow batches, such as the multi-token steps a speculative verify produces,
fuse the MoE up and gate lanes and size their mat-vec lane group from the
tensor shape and the device: a lane needs enough accumulation steps to cover
its reduction, and the grid that results still has to fill the compute
units. That puts them at or ahead of the canonical path per decode step,
worth about 6% on multi-token prediction with a 35B MoE; 2-8 token verify
batches take one mat-vec per expert assignment instead of the tiled GEMM
(+41% at four tokens on the 35B MoE). Perplexity is unchanged on MoE and
moves within floating-point reassociation on dense (6.7010 to 6.6858 on a
27B dense model at two tokens), while wide batches stay exact. Validated on
gfx906.

The fused up and gate mat-vec also admits Q4_K, Q5_K, Q6_K and IQ4_NL expert
weights, on by default, `GGML_CUDA_REPACK_KQUANT_MOE_FUSION=0` turns it off.
Qwen3.6-35B-A3B on one MI50, tg128 against the unfused path: Q4_K_M 80.9 ->
86.6 t/s (+7.1%), Q5_K_M 76.7 -> 83.1 (+8.4%), Q6_K 75.5 -> 81.2 (+7.5%),
IQ4_NL 84.8 -> 89.8 (+5.9%); two MI50 with `-sm layer` 74.5 -> 79.6 (+6.8%);
prefill unchanged; with `-sm tensor` the gain is within run-to-run spread
because the per-card slice shrinks while the AllReduce does not. Dense
models and Q8_0 experts are untouched (within 1%). The fused kernels are
bit-identical to the unfused ones at batch widths 2 to 4 and differ at
width 1 by reduction order: perplexity scored one token at a time over ten
2048-token chunks moves by 0.1 to 0.3% per type, inside the spread the
repack itself has against the canonical kernels. Multi-token prediction
output is unchanged. `GGML_CUDA_REPACK_MOE_FUSION_STATS=1` prints a
per-width histogram of fused launches.
