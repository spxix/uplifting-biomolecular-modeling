"""The DiT step-invariant hoist (``dit_hoist``) on the eager diffusion sampler, for a caller that keeps CUDA graphs off.

The kit installs ``dit_hoist`` together with its sampler CUDA graph (``fpf_clisampler``, ``PTX_SAMPLER_GRAPH=1``). A caller that keeps the
graphs off (its weights move between jobs, so captured addresses cannot follow them) calls :func:`install` once, after the mode's activation
and the model's construction, and :meth:`EagerHoist.release` when it moves the weights off the device. Every sampler call then goes through
the kit's admission rule (``fpf_clisampler.policy.admit`` with the mode's ``PTX_SAMPLER_ADMIT`` word, against the projected sampler-phase
peak), which chooses the hoist or the stock sampler. On the hoist route the step-invariant tensors of the denoiser (the token pair-bias
chain, the conditioning and atom-transformer terms) are recorded at the first denoiser step and served for the remaining steps; they are
the tensors the stock ops compute, so the sampler's outputs are unchanged. The buffers are released when the sampler returns, so nothing
rides into the confidence head or the next job. ``routes`` counts the admitted routes.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("protenix_opt.eager_hoist")

ROUTES = ("hoist_eager", "stock")


class _Sampler:
    """The object the policy functions drive: the hoist bound as ``biascache``."""

    biascache = None


class EagerHoist:
    def __init__(self, model):
        import dit_hoist
        import protenix.model.protenix as PM
        from fpf_clisampler import policy

        self.policy = policy
        self.sampler = _Sampler()
        # sampler= : the hoist's own sample_diffusion wrapper marks the sampler call, and the policy's run_hoist_eager drives
        # record / hit inside it.
        self.hoist = dit_hoist.install(model, sampler=self.sampler)
        self.routes = dict.fromkeys(ROUTES, 0)
        stock = PM.sample_diffusion

        def sample_diffusion(**kwargs):
            return self.sample(stock, **kwargs)

        sample_diffusion.__wrapped__ = stock
        PM.sample_diffusion = sample_diffusion

    def sample(self, stock, **kwargs):
        import torch

        policy = self.policy
        n_tok = int(kwargs["s_inputs"].shape[-2])
        n_atom = int(kwargs["input_feature_dict"]["atom_to_token_idx"].size(-1))
        n_sample = int(kwargs.get("N_sample") or 5)
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        admission = policy.admit(
            n_tok=n_tok,
            n_atom=n_atom,
            allocated_gib=torch.cuda.memory_allocated() / 2**30,
            trunk_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            total_gib=total,
            candidates=ROUTES,
            n_sample=n_sample,
        )
        self.routes[admission.route] += 1
        started = time.time()
        try:
            if admission.route == "hoist_eager":
                return policy.run_hoist_eager(self.sampler, stock, **kwargs)
            return policy.run_stock(self.sampler, stock, **kwargs)
        finally:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            at_exit = torch.cuda.memory_allocated() / 2**30
            released = self.release()
            policy.note_measured(
                admission, peak, n_tok=n_tok, n_atom=n_atom, alloc_at_exit_gib=at_exit,
                sampler_wall_s=time.time() - started, released=released,
                alloc_after_release_gib=torch.cuda.memory_allocated() / 2**30,
            )

    def release(self) -> str:
        """Drop the hoist's buffers and its padding-bias cache."""
        hoist = self.hoist
        hoist.set_mode("off")
        slots = sum(len((entry.get("dit_hoist") or {}).get("slots", {})) for entry in hoist.entries.values())
        hoist.release()
        padding = len(hoist._padbias)
        hoist._padbias.clear()
        return f"released={slots} g2_cleared={padding}"


def install(model) -> EagerHoist:
    """The eager hoist on ``model``'s sampler (once per process, after the mode's activation)."""
    hoist = EagerHoist(model)
    logger.info("dit_hoist on the eager sampler: %s", hoist.policy.words())
    return hoist
