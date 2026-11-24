"""Offline SHA256SUMS checks for the kit's shipped binaries and runtime loader contract tests (presence/ABI/symbols, zero runtime SHA256 or GPU probes)."""
import ctypes
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import types

import pytest

from opt_core import gates
from protenix_opt import binary_sums, stack

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # opt/
KIT_HOME = os.path.abspath(stack.kit_home())                                                   # opt/forward/flashpairformer
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")                                  # == the core tool's
SHIPPED = (                                                                                     # the binaries the kit ships today, relative to opt/
    "forward/flashpairformer/third_party/fastln_prebuilt/fast_layer_norm_cuda_v2_stream.so",
    "forward/flashpairformer/third_party/fastln_prebuilt_cu130/fast_layer_norm_cuda_v2_stream.so",
    "forward/flashpairformer/third_party/protenix_fpf_triatt_procuda/prebuilt/libtriatt_procuda_sm90a.so",
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _binaries():
    out = []
    for root, dirs, files in os.walk(OPT):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__", ".venv") and not d.endswith(".egg-info"))
        out += [os.path.relpath(os.path.join(root, f), OPT).replace(os.sep, "/") for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fresh(monkeypatch):
    """Caches cleared for each test."""
    monkeypatch.setattr(binary_sums, "_VERDICTS", {})
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])


@pytest.fixture
def copy(tmp_path):
    """A copy of the prologue library's prebuilt directory (binary + SHA256SUMS + manifest) to alter freely."""
    src = os.path.join(OPT, os.path.dirname(SHIPPED[2]))
    dst = tmp_path / "prebuilt"
    shutil.copytree(src, str(dst))
    return str(dst), os.path.basename(SHIPPED[2])


# ----------------------------------------------------------------------------------------------------------- offline SHA256SUMS checks

def test_the_kit_ships_exactly_the_known_binaries():
    assert tuple(_binaries()) == SHIPPED


def test_every_shipped_binary_is_a_line_of_the_sha256sums_beside_it_with_its_digest():
    problems = []
    for rel in SHIPPED:
        p = os.path.join(OPT, rel)
        sums = os.path.join(os.path.dirname(p), gates.BINARY_SUMS)
        if not os.path.isfile(sums):
            problems.append(f"{rel}: no {gates.BINARY_SUMS} beside it"); continue
        want = gates.read_sums(sums).get(os.path.basename(p))
        if want is None:
            problems.append(f"{rel}: not listed in {os.path.relpath(sums, OPT)}")
        elif want != _sha256(p):
            problems.append(f"{rel}: sha256 differs from {os.path.relpath(sums, OPT)}")
    assert not problems, problems


def test_every_sha256sums_beside_a_binary_is_in_sha256sum_format_and_lists_only_present_files():
    for rel in SHIPPED:
        d = os.path.dirname(os.path.join(OPT, rel))
        with open(os.path.join(d, gates.BINARY_SUMS), encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        assert lines, d
        for ln in lines:
            m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", ln)
            assert m, (d, ln)
            assert os.path.isfile(os.path.join(d, m.group(2))), (d, ln)


def test_the_sums_agree_with_the_loaders_own_manifests():
    """The SHA256SUMS line is the digest the package's manifest.json already pins (so_sha256): one fact, stated twice, equal."""
    for rel in SHIPPED:
        d = os.path.dirname(os.path.join(OPT, rel))
        with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        assert man["so"] == os.path.basename(rel)
        assert gates.read_sums(os.path.join(d, gates.BINARY_SUMS))[man["so"]] == man["so_sha256"]


# ----------------------------------------------------------------------------------------------------------- runtime presence-only checks

def test_binary_refusal_checks_presence_only_without_hashing(fresh, copy, capsys, monkeypatch):
    monkeypatch.setattr(gates, "sha256_file", lambda *a, **k: pytest.fail("runtime binary_refusal must not hash files"))
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("runtime binary_refusal must not call hashlib.sha256"))

    d, name = copy
    so = os.path.join(d, name)
    # Zeroed SHA256SUMS or absent SHA256SUMS does not refuse an existing binary at runtime
    with open(os.path.join(d, gates.BINARY_SUMS), "w", encoding="utf-8") as fh:
        fh.write("0" * 64 + "  " + name + "\n")
    assert binary_sums.binary_refusal(so) is None
    assert binary_sums.manifest_binary_refusal(d) is None

    os.remove(os.path.join(d, gates.BINARY_SUMS))
    assert binary_sums.binary_refusal(so) is None
    assert binary_sums.manifest_binary_refusal(d) is None

    # Missing file is refused by name
    os.remove(so)
    assert binary_sums.binary_refusal(so) == f"missing: {os.path.abspath(so)}"
    assert binary_sums.manifest_binary_refusal(d) is None

    os.remove(os.path.join(d, "manifest.json"))
    assert binary_sums.manifest_binary_refusal(d) is None
    assert capsys.readouterr().err == ""


# ----------------------------------------------------------------------------------------------------------- runtime loader contracts

def test_fastln_prebuilt_checks_abi_and_symbols_without_hashing_or_gpu_probes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    fp = _load_module_from_path("fastln_prebuilt_under_test", os.path.join(KIT_HOME, "third_party", "fastln_prebuilt.py"))
    monkeypatch.setattr(fp, "_sha", lambda *a, **k: pytest.fail("fastln_prebuilt must not hash files at runtime"))
    monkeypatch.setattr(fp, "_kernel_source_shas", lambda *a, **k: pytest.fail("fastln_prebuilt must not hash kernel sources at runtime"))

    # Stub protenix.model.layer_norm.layer_norm in sys.modules (no protenix package required)
    orig_ext = types.SimpleNamespace(forward_affine=lambda *a, **k: None, forward=lambda *a, **k: None)
    ln_mod = types.ModuleType("protenix.model.layer_norm.layer_norm")
    ln_mod.fast_layer_norm_cuda_v2 = orig_ext
    for pkg_name in ("protenix", "protenix.model", "protenix.model.layer_norm"):
        monkeypatch.setitem(sys.modules, pkg_name, types.ModuleType(pkg_name))
    monkeypatch.setitem(sys.modules, "protenix.model.layer_norm.layer_norm", ln_mod)

    # Guard against importing fastln_stream (no GPU probes on prebuilt load path)
    fs_guard = types.ModuleType("infopt_graphs.protenix.fastln_stream")
    fs_guard.verify_bitwise = lambda *a, **k: pytest.fail("fastln_prebuilt must not run verify_bitwise")
    fs_guard.verify_side_stream = lambda *a, **k: pytest.fail("fastln_prebuilt must not run verify_side_stream")
    monkeypatch.setitem(sys.modules, "infopt_graphs.protenix.fastln_stream", fs_guard)

    pre = tmp_path / "fastln_pre"
    pre.mkdir()
    monkeypatch.delenv("INFOPT_FASTLN_STREAM", raising=False)

    # 1. Missing manifest -> refused
    r = fp.install_prebuilt_fastln(str(pre))
    assert not r["installed"] and "no manifest" in r["reason"]

    base_man = {
        "so": "fast_layer_norm_cuda_v2_stream.so",
        "module_name": "fast_layer_norm_cuda_v2_stream",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
        "so_sha256": "0" * 64,
        "source_sha256": {"missing.cu": "0" * 64},
        "patched_launches": 25,
    }
    (pre / "manifest.json").write_text(json.dumps(base_man))

    # 2. Missing .so -> refused
    r = fp.install_prebuilt_fastln(str(pre))
    assert not r["installed"] and r["checks"]["so_exists"] is False

    # 3. ABI mismatches (torch / cuda / python major.minor) -> refused
    (pre / base_man["so"]).write_bytes(b"stub-so")
    for field, bad_val, check_key in (
        ("torch", "0.0.0+fake", "torch_version"),
        ("cuda", "0.0", "cuda_version"),
        ("python", "2.7.18", "python_version"),
    ):
        (pre / "manifest.json").write_text(json.dumps(dict(base_man, **{field: bad_val})))
        r = fp.install_prebuilt_fastln(str(pre))
        assert not r["installed"] and r["checks"][check_key] is False, (field, r)

    # 4. Missing public symbol -> refused
    (pre / "manifest.json").write_text(json.dumps(base_man))
    stub_new = types.ModuleType(base_man["module_name"])
    stub_new.forward_affine = lambda *a, **k: None

    class _FakeLoader:
        def __init__(self, name, path):
            self.name = name
            self.path = path
        def create_module(self, spec):
            return stub_new
        def exec_module(self, module):
            pass

    monkeypatch.setattr(importlib.machinery, "ExtensionFileLoader", _FakeLoader)
    r = fp.install_prebuilt_fastln(str(pre))
    assert not r["installed"] and r["checks"]["symbols"] is False and "forward" in r["reason"]

    # 5. Matching ABI + symbols -> installed directly without SHA256 or GPU probes
    stub_new.forward = lambda *a, **k: None
    r = fp.install_prebuilt_fastln(str(pre))
    assert r["installed"] is True
    assert "bitwise" not in r and "side_stream" not in r
    assert ln_mod.fast_layer_norm_cuda_v2 is stub_new
    assert ln_mod._infopt_stream_patched is True

    # 6. Idempotent second call returns already installed without re-probing
    r2 = fp.install_prebuilt_fastln(str(pre))
    assert r2["installed"] is True and r2["reason"] == "already installed"


def test_triatt_procuda_binding_loads_without_sha256(tmp_path, monkeypatch):
    binding_path = os.path.join(KIT_HOME, "third_party", "protenix_fpf_triatt_procuda", "binding.py")
    B = _load_module_from_path("procuda_binding_under_test", binding_path)

    monkeypatch.setattr(B, "_LIB", None)
    monkeypatch.setattr(B, "PREBUILT_DIR", str(tmp_path))
    monkeypatch.setattr(B, "_sha256", lambda *a, **k: pytest.fail("binding.load must not hash .so at runtime"))
    monkeypatch.setattr(B, "source_sha256", lambda *a, **k: pytest.fail("binding.load must not hash source at runtime"))

    # Missing .so raises RuntimeError by name
    with pytest.raises(RuntimeError, match="prebuilt binary missing"):
        B.load()

    # Present .so loads directly via ctypes.CDLL without SHA256 checks
    so = tmp_path / B.SO_NAME
    so.write_bytes(b"stub-so")
    loaded_paths = []
    fake_lib = types.SimpleNamespace(f1v2_launch=types.SimpleNamespace(), f1v2_smem_bytes=types.SimpleNamespace())

    def _fake_cdll(path):
        loaded_paths.append(path)
        return fake_lib

    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
    monkeypatch.setattr(B.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(B.torch.cuda, "get_device_properties", lambda *a: types.SimpleNamespace(multi_processor_count=132))
    assert B.load() is fake_lib
    assert loaded_paths == [str(so)]


def test_deadskip_install_skips_checkpoint_hashing_and_defaults_reprobe_false(monkeypatch):
    torch = pytest.importorskip("torch")
    ds = _load_module_from_path("deadskip_under_test", os.path.join(KIT_HOME, "src", "deadskip.py"))

    assert inspect.signature(ds.install).parameters["reprobe"].default is False
    monkeypatch.setattr(ds, "sha256_file", lambda *a, **k: pytest.fail("deadskip.install must not hash checkpoints at runtime"))
    monkeypatch.setattr(ds, "probe", lambda *a, **k: pytest.fail("deadskip.install must not run GPU probe when reprobe=False"))

    class _DummyMod(torch.nn.Module):
        def __init__(self, val: float):
            super().__init__()
            self.w = torch.nn.Parameter(torch.full((2, 2), val))

        def forward(self, x, *args, **kwargs):
            return x + self.w.sum()

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dead = _DummyMod(1e-7)
            self.live = _DummyMod(0.5)

    model = _Root()
    man = {
        "checkpoint_sha256": "0" * 64,
        "thr_absmax": 1e-4,
        "probe_sizes": [16],
        "modules": [
            {"path": "dead", "class": "_DummyMod", "kind": "pair", "probe_pass": True},
            {"path": "live", "class": "_DummyMod", "kind": "pair", "probe_pass": True},
        ],
    }

    skipped = ds.install(model, man, ckpt_path="/nonexistent/checkpoint.pt")
    assert skipped == ["dead"]
    reasons = {r["path"]: r for r in ds.install.last_report}
    assert reasons["dead"]["action"] == "SKIPPED"
    assert reasons["live"]["action"] == "STOCK" and reasons["live"]["reason"] == "WEIGHTS_CHANGED"
    ds.remove(model)


def test_stack_dry_run_skips_graphed_py_sha256(monkeypatch):
    monkeypatch.setattr(stack, "graphed_py_sha256", lambda *a, **k: pytest.fail("_dry_run must not hash graphed.py at runtime"))
    monkeypatch.setattr(stack._report, "log_activation", lambda *a, **k: None)
    monkeypatch.setattr(stack, "_base", lambda mode, trigger=None: {
        "mode": mode, "trigger": trigger, "gpu": {"sm": "9.0", "name": "stub"}, "notes": [], "_cc": "9.0",
    })
    monkeypatch.setattr(stack, "_gates", lambda base: None)
    monkeypatch.setattr(stack, "cells", lambda sm: {})
    fake_res = types.SimpleNamespace(
        row="exact", row_key="exact", base="exact", pre_exports={}, exports={}, unsets=[],
        extras=[], kernel_key="k", kit_spec="s", notes=[], levers=[], pythonpath="",
        final=lambda env: dict(env),
    )
    monkeypatch.setattr(stack, "resolve", lambda *a, **k: fake_res)
    monkeypatch.setattr(stack, "_classify", lambda *a, **k: ([], [], [], {}))
    monkeypatch.setattr(stack, "_row_levers", lambda *a, **k: [])
    rep = stack._dry_run("exact", strict=False)
    assert rep["graphed_py_sha256"] is None
