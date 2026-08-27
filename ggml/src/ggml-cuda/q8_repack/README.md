# Repacked-weight path (AMD GCN)

Custom matmul frontend for gfx906 (Q8_0 and MXFP4). Weights are converted at upload into a
two-plane layout consumed by dp4a-based mat-vec and tiled GEMM kernels. Kernel
bodies are guarded by `#if defined(GGML_USE_HIP) && defined(__gfx906__)`;
elsewhere they emit `NO_DEVICE_CODE` stubs.

The folder is self-contained: the only header seen outside is `repack.cuh`
(included by `ggml-cuda.cu`). It exposes three integration channels:

1. **Weight upload** - the repack buffer type (`buffer.cu`) is offered to the
   loader as an extra buffer type; `set_tensor` converts supported Q8_0 tensors
   into the repacked layout on upload.
2. **Compute dispatch** - `ggml_cuda_mul_mat{,_id}` route to the repacked
   kernels when `ggml_cuda_repack_mul_mat_should_fire(src0)` is true.
3. **Fused FFN / bias-only dispatch** - the repacked MMV implements fused
   epilogues (dual accumulator + GLU / `+bias`), so `{MM, MM, GLU}`,
   `{MM_ID, MM_ID, GLU}` and `{MM, ADD}` subgraphs are fused for repacked
   weights via explicit branches in `ggml_cuda_try_fuse`. The canonical-layout
   fused kernels stay suppressed.

## The repacked weight layout

A Q8_0 matrix with `ne0` columns and `ne1` rows is stored as two planes:

```
qs row stride (bytes):  ne0, +16 when that is a multiple of 128
                        (keeps row starts off the HBM channel stride; 16 keeps
                         uint4 alignment)

qs plane  [ne1 * qs_stride bytes]  quantized int8 values, sub-block 32 bytes
d plane   [ne1 * ne0/32 * 2 bytes] f16 scale per 32-value sub-block
```

- Scales keep their own plane: the narrow mat-vec holds several rows per lane
  and reads their scales together, so they must stay adjacent.
- Layout math lives in `repack_qs_stride()` / `repack_gcn_nbytes()` /
  `repack_q8_0_host()`. Reads back to canonical `block_q8_0` are unsupported:
  weights are write-once, `get_tensor` is nullptr.
- Multi-expert weights (`ne2 > 1`) concatenate one such layout per expert; the
  per-expert stride is `repack_gcn_nbytes(type, ne0, ne1)`, passed as
  `expert_stride`.

## Compute paths

Three kernel families, chosen per 2D slice:

| Path | Kernel | Trigger |
|---|---|---|
| Mat-vec (single token) | `mul_mat_vec_q8_0_repacked<ROWS,NWAVES,HAS_IDS,LANES>` | `ne11 == 1` (dense) / `n_tokens == 1` (MoE) |
| Mat-vec (narrow batch) | `mul_mat_vec_repacked_nc<ROWS,NWAVES,NCOLS,RPL>` | `2 <= ne11 <= MMQ_RP_Q8_MMV_MAX_TOKENS`, dense only |
| GEMM 64-wide | `mmq_gemm_repacked<HAS_IDS,TN_,NRL>` | `ne11 >= 128` (dense) / `n_assign >= 2*BN*n_expert` (MoE) |
| GEMM 32-wide | `mmq_gemm_repacked_w32<HAS_IDS,TN_,NRL>` | `MMQ_RP_Q8_MMV_MAX_TOKENS < ne11 < 128` (dense) / `n_assign < 2*BN*n_expert` (MoE) |

The narrow mat-vec is dense-only. MoE keeps the tile from two tokens up: its
token counts are per expert, so a narrow ubatch does not imply a narrow tile.

Both GEMM wrappers forward to one device function
`mmq_gemm_repacked_impl<HAS_IDS, CW, TN_, NRL>`.

**Dense vs MoE.** The kernels are single templates parameterized on `HAS_IDS`;
only the host orchestration differs. Dense (`mul-mat.cu`) quantizes src1,
splits into 2D slices, `blockIdx.y` = token tile. MoE (`mul-mat-id.cu`) routes
tokens via `ggml_cuda_launch_mm_ids_helper`, prefix-sums tiles with
`repack_tile_off` into `tile_off` / `tile_meta`, and `blockIdx.y` is a flat
tile id decoded to `(expert, a_base)` via `tile_meta` (O(1), no search).

**MMV.** Each wavefront handles `ROWS` output rows (`WPR = NWARPS/ROWS` warps
per row; current instantiations use `WPR=1`). `LANES` lanes split each 32-value
sub-block into two 16-value halves (`n_half = 2 * n_blocks`) and stride over
them; per element a `dp4a` over 4 int32 words, scaled by `dw * dx`, reduced
with `rp_warp_reduce_sum<LANES>` (DPP intrinsics on GCN, `warp_reduce_sum`
elsewhere). `LANES` = 64 for dense, 16 for MoE, 32 for the fused MoE MMV.

**GEMM.** Tile is `BM` rows x `BN = CW x TN_` cols x `BK` sub-blocks deep. Data
flows global -> registers (prefetch) -> shared (`sW_lo/sW_hi`, weight scales
`sWdh` k-major, input `sXq` swizzled via `sX_swizzle<CW>`, `sXd`) -> registers
(`acc[NROW][TN_]`). The K loop is software-pipelined (double-buffered). Two
load paths: **full** (unchecked, only when the whole tile is in-bounds) and
**checked** (bounds-tested per element). The epilogue vectorizes with `float4`
when the output row stride is 4-aligned; MoE scatters via `ids_dst`.

**Multi-token input format.** This applies to the tiled GEMM only; the narrow
mat-vec takes plain `block_q8_1` rows, the same layout as the single-token
path, with a row stride of `ne10_padded / QK8_1` blocks.
src1 is quantized with the grouped MMQ layout
(`quantize_mmq_q8_1_cuda`). The buffer is reinterpreted as `block_q8_1_mmq_h`,
which exposes the same 64 bytes as `float d4[4]` instead of the stock `half2`
union, so activation scales read directly as f32 (no extra fp16->fp32 pass).
Three `static_assert`s pin size and field offsets to the stock layout.

**Fused MMV (repacked FFN / bias-only).** Single token only, dense `ne[1] == 1`
and MoE `ne[2] == 1`. Anything wider, including every step of a speculative
verify batch and every multi-slot decode, falls through to the narrow mat-vec
or the GEMM:

| Fusion | Pattern | Kernel instantiation |
|---|---|---|
| GLU | `{MM, MM, GLU}` | `<16,16,false,64,true>` (gate lane) |
| bias-only | `{MM, ADD}` | `<16,16,false,64,true>` (no gate lane) |
| MoE GLU | `{MM_ID, MM_ID, GLU}` | `<8,4,true,32,true>` (gate lane) |

Template signature: `<ROWS, NWAVES, HAS_IDS, LANES, HAS_FUSION>`.

- Both GLU lanes share the same quantized input `xq`; fusing only doubles the
  weight-side loads (`wbase_gate`, `d_row_gate`).
- `HAS_FUSION=true` enables a dual accumulator `acc_gate` plus the fused
  epilogue (`rp_mmv_fusion_epilogue`). A runtime `use_gate = wbase_gate !=
  nullptr` selects GLU vs bias-only within the same instantiation, so `{MM,
  ADD}` never dereferences a null gate pointer.
- The plain (non-fused) MMV dispatch uses the template default
  `HAS_FUSION=false`; all gate code is stripped at compile time via
  `if constexpr`.
- Bias indexing: dense `row`, MoE `expert * ne1 + row`.
- The GLU branch fires only when **both** lanes are repacked
  (`should_fire(src0) && should_fire(gate->src[0])`), so a mixed up-repacked /
  gate-canonical state can never reach a kernel that reads `gate->data` as
  repacked layout.
- MoE fusion covers the plain `{MM_ID, MM_ID, GLU}` shape only. Per-expert bias
  (`{MM_ID, ADD_ID, MM_ID, ADD_ID, GLU}`) and the per-expert scale subgraph are
  separate branches in `ggml_cuda_try_fuse` and are not wired up.
- With ids, the gate base is advanced per routed expert alongside `wbase`.
  Without that every assignment reads expert 0's gate weights.
- MoE geometry is `<8,4,32>`, not the dense `<16,16,64>`. A 1024-thread
  workgroup is 16 waves, 4 per SIMD, which caps registers at 64 per lane, and a
  fused MoE row does not fit that: traced against up and gate as separate
  kernels, MoE mat-vec time per token is +3.5% at `<64,16,16>` and -15.9% at
  `<8,4,32>`.

## Files

| File | Contents | Used by |
|---|---|---|
| `repack.cuh` | Public API (7 functions: buft predicate, buffer-type factory, tensor support, should-fire, dense/MoE/mat-vec-fused dispatch). The only header included outside the folder. | `ggml-cuda.cu` |
| `repack-common.cuh` | Tuning knobs (`MMQ_RP_Q8_*`), layout math (`repack_qs_stride`, `repack_gcn_nbytes`), X swizzle (`sX_swizzle<CW>`), device structs (`rp_x_sub`, `block_q8_1_mmq_h`, `sXq_row_q8`), input-gather helpers, DPP warp reduce, host helper declarations. | all TUs in folder |
| `repack-kernels.cuh` | All device kernels: `repack_tile_off`, `mul_mat_vec_q8_0_repacked` (optional `HAS_FUSION` epilogue), `mmq_gemm_repacked_impl` + 64/32-wide launch wrappers, `repack_tile_meta`. GCN-guarded, `NO_DEVICE_CODE` elsewhere. | `mul-mat.cu`, `mul-mat-id.cu` |
| `repack-common.cu` | `ggml_cuda_repack_tensor_supported()`, `ggml_cuda_repack_mul_mat_should_fire()` (also handles views), host repack `repack_q8_0_host()`, persistent per-view cache `repack_view_get_cached`. | `mul-mat.cu`, `mul-mat-id.cu`, `buffer.cu`, `ggml-cuda.cu` |
| `mul-mat.cu` | Dense entry `ggml_cuda_mul_mat_repacked()` + per-slice dispatcher, dense fused host `ggml_cuda_mul_mat_vec_repacked_fused()`. | `ggml-cuda.cu` |
| `mul-mat-id.cu` | MoE entry `ggml_cuda_mul_mat_id_repacked()` (token routing, tile prefix-sum, GEMM dispatch). | `ggml-cuda.cu` |
| `buffer.cu` | Repack buffer type: `set_tensor` (repacks supported Q8_0, handles full and staged partial writes), `get_alloc_size`, factory `ggml_backend_cuda_repack_buffer_type()`, `ggml_backend_buft_is_cuda_repack()`. `get_tensor` is nullptr. | backend registration |

External deps (parent `ggml-cuda/`): `common.cuh` (dp4a, warp_reduce_sum),
`mmq.cuh` (`block_q8_1_mmq`, `QK8_1_MMQ`), `quantize.cuh`
(`quantize_{row,mmq}_q8_1_cuda`), `mmid.cuh` (MoE helper), `unary.cuh` (GLU
ops), `ggml-backend-impl.h` (buffer struct / backend interface access in
`buffer.cu`).

## Persistent view cache

MoE expert slices are views of a repacked tensor; views are not valid in
repacked layout (the scale-plane offset depends on the full `ne1`), so each
unique view is re-packed into a dedicated buffer. These are `cudaMalloc`'d
once per unique view and never freed (pool allocs are reused across graph
evaluations, which breaks captured CUDA graphs). Keyed by `(view->data, ne0,
ne1, ne2)`; guarded by a mutex; shared by dense and MoE. At creation the QS
plane is copied from the view's offset in the base QS plane and the scale plane
from the matching offset in the base scale plane; hits just return the pointer.

## Build and integration

- Sources are collected by `file(GLOB ... "q8_repack/*.cu" / "q8_repack/*.cuh")`
  in **both** `ggml-cuda/CMakeLists.txt` and `ggml-hip/CMakeLists.txt`. Adding or
  renaming files requires a reconfigure.
- Enabled only on gfx906 (the arch the kernels compile for); elsewhere the
  buffer-type factory returns nullptr and the plain CUDA buffer type is used.
  Runtime control follows the CPU weight repack: extra buffer types carry the
  repack buft, so `--no-repack` (`use_extra_bufts = false`) disables it and no
  repack structural code is active.

Integration points in `ggml-cuda.cu` (the only TU outside the folder that sees
`repack.cuh`):

| Site | Hook |
|---|---|
| top of file | `#include "ggml-cuda/q8_repack/repack.cuh"` |
| `get_extra_bufts` | `ggml_backend_cuda_repack_buffer_type(dev)` - expose the repack buft to the loader |
| `device_supports_buft` | `ggml_backend_buft_is_cuda_repack(buft)` - claim repacked tensors |
| `device_supports_op` | reject a repacked weight unless it is `src0` of `MUL_MAT` / `MUL_MAT_ID` with f32 src1/dst; accepts a view whose base is supported |
| `ggml_cuda_mul_mat()` | should-fire -> `ggml_cuda_mul_mat_repacked()` |
| `ggml_cuda_mul_mat_id()` | should-fire -> `ggml_cuda_mul_mat_id_repacked()` |
| `ggml_cuda_try_fuse` GLU site | dense single-token `{MM, MM, GLU}` -> `ggml_cuda_mul_mat_vec_repacked_fused()` |
| `ggml_cuda_try_fuse` ADD site | dense `{MM, ADD}` (no gate lane) -> `ggml_cuda_mul_mat_vec_repacked_fused()` |
| `should_fuse_mul_mat_vec_{f,q}()` | should-fire -> `false` (suppress canonical fused MMV) |

The TP wrapper (`ggml-backend-meta.cpp`) plies the repack buft through the same
channels: `get_extra_bufts` composes per-lane repack bufts into a meta repack
buft (tagged by a `repack` flag in the meta buft context; the predicate stays
static inside the meta TU, so no ggml API is extended), the meta `supports_op`
mirrors the CUDA gate, and `set_tensor_async` bypasses chunk-splicing upload so
repacked data reaches `set_tensor` intact. The loader assigns weights by
iterating the buft list in order, and `make_gpu_buft_list` places extra bufts
before the device default, so supported Q8_0 weights land in the repack buft.

Dispatch, upload, and suppression all key off one predicate,
`ggml_cuda_repack_mul_mat_should_fire(src0)`, which also accepts views
(`src0->view_src`). A weight is either fully in the repacked layout or fully
canonical; the repack fused branches sit before the canonical `_f`/`_q` checks
so a repacked weight always takes the repacked kernel.

Cost: the `HAS_FUSION=true` instantiations use more registers and lower
occupancy than the plain `HAS_FUSION=false` ones. Fresh VGPR/SGPR/occupancy
numbers land in `compiled_kernel_stats/`.

## Known gaps

- MoE fused GLU covers only the no-bias `{MM_ID, MM_ID, GLU}` shape, and only at
  one token. Bias and scale subgraphs, and every batch wider than one token, use
  the unfused path.
- Q8_0 and MXFP4 are supported by the layout machinery.

