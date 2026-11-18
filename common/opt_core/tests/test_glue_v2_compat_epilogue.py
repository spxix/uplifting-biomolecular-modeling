"""V2_EPILOGUE_BACKEND: which statement serves fpf_glue_v2.kernels.epilogue_v3 (CPU, synthetic tensors)."""
import types

import pytest

torch = pytest.importorskip("torch")

from opt_core.kernels import fpf_glue_v2_compat as compat  # noqa: E402

I, J, H, D, C = 6, 5, 2, 8, 12


def operands(seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn((1, I, H, J, D), generator=g).to(torch.bfloat16)          # cuEquivariance layout
    gate = torch.randn((1, I, J, H * D), generator=g).to(torch.bfloat16)
    wo = (torch.randn((C, H * D), generator=g) * 0.2).to(torch.bfloat16)
    return o, gate, wo


def reference(o, gate, wo):
    """The epilogue statement with its rounding points: sigmoid(g), o * g, the projection in bf16."""
    sg = torch.sigmoid(gate[0].float()).to(torch.bfloat16)
    og = (o[0].permute(0, 2, 1, 3).reshape(I, J, H * D).float() * sg.float()).to(torch.bfloat16)
    return (og.float() @ wo.float().t()).to(torch.bfloat16)                   # [I, J, C]


def kernels_module(calls):
    def epilogue_v3(o, g, wo16, z=None, **kwargs):
        calls.append(kwargs)
        return "kernel"

    return types.SimpleNamespace(epilogue_v3=epilogue_v3)


def test_backend_word_is_validated(monkeypatch):
    monkeypatch.delenv(compat.EPILOGUE_BACKEND_ENV, raising=False)
    assert compat.epilogue_backend() == "kernel"
    monkeypatch.setenv(compat.EPILOGUE_BACKEND_ENV, "cuBLAS")
    assert compat.epilogue_backend() == "cublas"
    monkeypatch.setenv(compat.EPILOGUE_BACKEND_ENV, "torch")
    with pytest.raises(ValueError, match="V2_EPILOGUE_BACKEND"):
        compat.epilogue_backend()


@pytest.mark.parametrize("ending", [False, True])
def test_cublas_backend_updates_the_residual_in_place(monkeypatch, ending):
    monkeypatch.setenv(compat.EPILOGUE_BACKEND_ENV, "cublas")
    calls = []
    module = kernels_module(calls)
    compat.patch_epilogue(module)
    o, gate, wo = operands()
    z0 = torch.randn((1, J, I, C) if ending else (1, I, J, C)).to(torch.bfloat16)
    z = z0.clone()
    assert module.epilogue_v3(o, gate, wo, z, ending=ending, residual=True, cfg={}, woT16=wo.t().contiguous()) is None
    assert calls == []
    u = reference(o, gate, wo)
    expected = (z0[0].float() + (u.transpose(0, 1) if ending else u).float()).to(torch.bfloat16)
    torch.testing.assert_close(z[0], expected, rtol=1e-2, atol=1e-2)


def test_cublas_backend_returns_the_projection(monkeypatch):
    monkeypatch.setenv(compat.EPILOGUE_BACKEND_ENV, "cublas")
    calls = []
    module = kernels_module(calls)
    compat.patch_epilogue(module)
    o, gate, wo = operands(1)
    out = torch.empty((1, I, J, C), dtype=torch.bfloat16)
    result = module.epilogue_v3(o, gate, wo, None, out=out, cfg={}, woT16=wo.t().contiguous())
    assert calls == [] and result.shape == (1, I, J, C) and result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result[0], reference(o, gate, wo), rtol=1e-2, atol=1e-2)


def test_calls_the_cublas_path_does_not_serve_run_the_kernel(monkeypatch):
    monkeypatch.setenv(compat.EPILOGUE_BACKEND_ENV, "cublas")
    calls = []
    module = kernels_module(calls)
    compat.patch_epilogue(module)
    o, gate, wo = operands()
    batched = (o.expand(2, *o.shape[1:]), gate.expand(2, *gate.shape[1:]))
    assert module.epilogue_v3(*batched, wo, None, cfg={}) == "kernel"
    assert module.epilogue_v3(o.float(), gate, wo, None, cfg={}) == "kernel"
    assert len(calls) == 2


def test_kernel_backend_keeps_the_kernel(monkeypatch):
    monkeypatch.delenv(compat.EPILOGUE_BACKEND_ENV, raising=False)
    calls = []
    module = kernels_module(calls)
    compat.patch_epilogue(module)
    o, gate, wo = operands()
    assert module.epilogue_v3(o, gate, wo, None, cfg={}) == "kernel" and len(calls) == 1
