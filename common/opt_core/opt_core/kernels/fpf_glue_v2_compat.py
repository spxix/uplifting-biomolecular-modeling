"""Explicit compatibility changes applied outside pinned upstream source files.

``V2_EPILOGUE_BACKEND`` selects what serves ``fpf_glue_v2.kernels.epilogue_v3`` (the
tri-attention epilogue: sigmoid gate, o * g, output projection, residual add):

  kernel  the kit's Triton kernel (default);
  cublas  the same statement as PyTorch ops with the projection on cuBLAS. Its rounding
          points are the kernel's (sigmoid(g) and o * g in bf16, the projection accumulated
          in fp32 and rounded to bf16, the residual add in bf16), so outputs are equal to
          the kernel's. Calls it does not serve (batched o / g, other dtypes or layouts) run
          the kernel.
"""

import functools
import importlib.abc
import os
import sys

EPILOGUE_BACKEND_ENV = "V2_EPILOGUE_BACKEND"
EPILOGUE_BACKENDS = ("kernel", "cublas")
STATS = {"backend": None, "cublas": 0, "kernel": 0}


def epilogue_backend(environ=None) -> str:
    environ = os.environ if environ is None else environ
    value = (environ.get(EPILOGUE_BACKEND_ENV) or "kernel").strip().lower()
    if value not in EPILOGUE_BACKENDS:
        raise ValueError(f"{EPILOGUE_BACKEND_ENV}={value!r} is not one of {'|'.join(EPILOGUE_BACKENDS)}")
    return value


def cublas_epilogue(o, g, wo16, z=None, *, ending=False, residual=False, out=None, woT16=None):
    """``epilogue_v3`` on cuBLAS for one item, or ``NotImplemented`` when the call is not one it serves.

    o: [1, I, H, J, D] or [I, H, J, D] (cuEquivariance layout), g: [.., I, J, H * D],
    wo16: [C, H * D]. With ``residual`` z ([.., I, J, C]; [.., J, I, C] with ``ending``) is
    updated in place and None is returned, else the projection [.., I, J, C].
    """
    import torch

    bf16 = torch.bfloat16
    o5 = o if o.dim() == 5 else (o.unsqueeze(0) if o.dim() == 4 else None)
    g4 = g if g.dim() == 4 else (g.unsqueeze(0) if g.dim() == 3 else None)
    if o5 is None or g4 is None or o5.shape[0] != 1 or g4.shape[0] != 1:
        return NotImplemented
    _, I, H, J, D = o5.shape
    C = wo16.shape[0]
    if (o.dtype != bf16 or g.dtype != bf16 or wo16.dtype != bf16 or o5.stride(-1) != 1
            or tuple(g4.shape) != (1, I, J, H * D) or not g4.is_contiguous()
            or tuple(wo16.shape) != (C, H * D)):
        return NotImplemented
    if residual:
        if z is None or z.dtype != bf16 or z.stride(-1) != 1:
            return NotImplemented
        z4 = z if z.dim() == 4 else (z.unsqueeze(0) if z.dim() == 3 else None)
        if z4 is None or tuple(z4.shape) != ((1, J, I, C) if ending else (1, I, J, C)):
            return NotImplemented
    gated = torch.sigmoid(g4[0].view(I, J, H, D))          # bf16 (fp32 opmath)
    gated.mul_(o5[0].permute(0, 2, 1, 3))                    # o * g -> bf16, token-major [I, J, H, D]
    lhs = gated.view(I * J, H * D)
    woT = _transposed_weight(wo16, woT16)                    # the kernel's [H * D, C] weight
    if residual:
        u = torch.mm(lhs, woT)                               # bf16 operands, fp32 accumulate -> bf16
        zz = z4[0].transpose(0, 1) if ending else z4[0]
        zz.add_(u.view(I, J, C))
        return None
    if out is not None and out.is_contiguous() and out.numel() == I * J * C and out.dtype == bf16:
        u = torch.mm(lhs, woT, out=out.view(I * J, C))
    else:
        u = torch.mm(lhs, woT)
        if out is not None:
            out.view(I * J, C).copy_(u)
            u = out
    return u.reshape(tuple(g.shape[:-1]) + (C,))


def _transposed_weight(wo16, woT16):
    if woT16 is not None:
        return woT16
    try:
        from fpf_triatt_epi.epilogue import _woT
    except ImportError:
        from opt_core.kernels.fpf_triatt_epi.epilogue import _woT
    return _woT(wo16)


def patch_epilogue(module):
    import torch
    import triton

    original = module.epilogue_v3
    if getattr(original, "_v2_compat", False):
        return
    backend = epilogue_backend()
    STATS["backend"] = backend
    count = 0

    @functools.wraps(original)
    def epilogue(o, g, wo16, z=None, **kwargs):
        nonlocal count
        if backend == "cublas":
            served = cublas_epilogue(
                o, g, wo16, z,
                ending=kwargs.get("ending", False), residual=kwargs.get("residual", False),
                out=kwargs.get("out"), woT16=kwargs.get("woT16"),
            )
            if served is not NotImplemented:
                STATS["cublas"] += 1
                if STATS["cublas"] == 1:
                    print("V2_COMPAT name=epilogue_backend kernel=epilogue_v3 backend=cublas", flush=True)
                return served
        STATS["kernel"] += 1
        cfg = kwargs.get("cfg")
        if (
            o.is_cuda
            and torch.cuda.get_device_name(o.device) == "NVIDIA H20"
            and torch.__version__ == "2.13.0+cu130"
            and triton.__version__ == "3.7.1"
            and cfg is not None
            and cfg.get("NUM_STAGES_K") == 3
        ):
            kwargs["cfg"] = dict(cfg, NUM_STAGES_K=1)
            count += 1
            if count == 1:
                print(
                    "V2_COMPAT name=h20_epilogue_single_stage kernel=epilogue_v3 from_stages=3 to_stages=1",
                    flush=True,
                )
        return original(o, g, wo16, z, **kwargs)

    epilogue._v2_compat = True
    module.epilogue_v3 = epilogue


class Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        patch_epilogue(module)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.resolving = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.resolving:
            return None
        if fullname not in {
            "fpf_glue_v2.kernels",
            "opt_core.kernels.fpf_glue_v2.kernels",
        }:
            return None
        self.resolving.add(fullname)
        try:
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None:
                        spec.loader = Loader(spec.loader)
                    return spec
        finally:
            self.resolving.remove(fullname)
        return None


def install():
    if os.environ.get("V2_COMPAT_EPILOGUE") != "1" and epilogue_backend() == "kernel":
        return
    if not any(isinstance(finder, Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, Finder())
