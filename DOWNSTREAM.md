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
