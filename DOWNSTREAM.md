# Downstream integration

Base: Anthropic `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
The original multi-model release, licenses, lookup tables and prebuilt artifacts
remain intact. Consumers pin a Git commit; they need not install every kit.

## H20 compatibility

`opt_core.kernels.fpf_glue_v2_compat.install()` is an opt-in import adapter.
With `V2_COMPAT_EPILOGUE=1`, it changes epilogue `NUM_STAGES_K=3` to `1` only
for NVIDIA H20, Torch 2.13.0+cu130 and Triton 3.7.1. It leaves the original
configuration object unchanged. Importing this module alone applies no patch.

The implementation was moved unchanged from Protenix commit
`89468322da16f708a928d571c386f5569d6883e6`, where the three L1 profiles were
validated against their frozen upstream coordinate references on H20.
This is a hardware compatibility adapter, not a new optimization profile.
Model/service activation remains the consumer's responsibility. InsFold model
extensions and checkpoint-specific policies do not belong in this adapter.

## Tri-attention epilogue backend

The same adapter reads `V2_EPILOGUE_BACKEND` (`kernel` by default, or `cublas`)
when `fpf_glue_v2.kernels` is imported; `install()` also adds the import hook
when only this variable is set. `cublas` runs the epilogue statement of
`epilogue_v3` (sigmoid gate and o·g rounded to bf16, the output projection on
cuBLAS with fp32 accumulation, the residual add in bf16) as PyTorch ops for
single-item bf16 calls; other calls keep the Triton kernel. Its outputs equal
the kernel's on PPU-ZW810E and H20 at 1024 and 1536 tokens, both nodes. It is
the faster statement on the PPU and the slower one on H20, so the consumer
selects it per device.

## Protenix upstream 4c355be

The host-indexed confidence summary also calls upstream's
`calculate_chain_pair_pae` when present, preserving its new mean/min fields.
The original 2475421 wheel pin remains the release provenance, not the identity
of a downstream source-built model. Consumers pin their own model source.

`LAYERNORM_TYPE=torch` under exact keeps the model's normalization calls.
MK-PF, CUDA prologue and XL block fusion that emulate fast LayerNorm are disabled
and recorded in the activation report. Other exact optimizations remain enabled.
Conflicting fusion overrides and unverified fast/big + Torch combinations fail
explicitly. The default fast_layernorm release modes are unchanged.

Sampler graph warmup waits for the current stream after creating `x0_keep` and
binding hoist buffers. Waiting before the clone did not order its producer with
the side-stream copy, allowing the first sample of a new shape to read stale
data under GPU contention. This change adds the missing stream dependency; it
does not change the sampler math or consume random numbers.

## Padded tri-attention buffer release

The PAD8 trunk path shared one padded q/k/v/bias/g set per padded token count
and never released it. A long-lived service process kept one set for every
distinct unaligned size it served (about 2080 x P^2 bytes; 13.6 GiB at 2560
tokens), so device memory grew across items until every later trunk ran out of
memory. Allocating a new P class now drops the other classes, and
`pred_release` drops the previous item's sets before the next predict, also
after an item that raised. A set used inside a CUDA-graph capture is pinned
and never released, because the captured graph replays its addresses. Released
sets are rebuilt zero-filled on their next use, so outputs are unchanged.

## Caller-owned cuEquivariance tile tables

`env.sh` sets `CUEQ_TRITON_CACHE_DIR` to the kit's `cueq_cache/` only when the
caller has not set it, so a consumer that points cuEquivariance at tile tables
of its own (for cards the kit ships no table for) keeps them in every kit mode;
the activation report notes the caller's directory. The stock route
(`pred --mode off`) keeps such a directory as well, and still refuses the
kit's `cueq_cache/`. Tables are a pure lookup of tile shapes per GEMM key.

## Eager DiT hoist

`protenix_opt.eager_hoist.install(model)` puts `dit_hoist` on the stock
(graph-free) sampler for a consumer that keeps the sampler CUDA graph off.
It is an explicit call after the mode's activation and the model's
construction. Each sampler call goes through `fpf_clisampler.policy.admit`
with the mode's `PTX_SAMPLER_ADMIT` word and runs either the hoist or the
stock sampler; the hoist's buffers are released when the sampler returns,
and `EagerHoist.release()` releases them again on demand. The hoisted tensors
are the ones the stock ops compute, so outputs are unchanged.
