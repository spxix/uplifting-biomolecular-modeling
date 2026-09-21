"""Explicit compatibility changes applied outside pinned upstream source files."""

import functools
import importlib.abc
import os
import sys


def patch_epilogue(module):
    import torch
    import triton

    original = module.epilogue_v3
    if getattr(original, "_v2_compat", False):
        return
    count = 0

    @functools.wraps(original)
    def epilogue(o, g, wo16, z=None, **kwargs):
        nonlocal count
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
    if os.environ.get("V2_COMPAT_EPILOGUE") != "1":
        return
    if not any(isinstance(finder, Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, Finder())
