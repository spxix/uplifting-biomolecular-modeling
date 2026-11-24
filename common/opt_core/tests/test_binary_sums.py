"""Offline SHA256SUMS tooling (tools/binary_sums.py) and runtime loader contract (presence + ABI checks, zero runtime SHA256 / GPU probes).

Offline: every compiled binary under opt_core/ (.so, .cubin, .cubin.xz) remains covered by a SHA256SUMS file verifiable via
tools/binary_sums.py --check.
Runtime: gates.binary_refusal() and kernel loaders perform structural presence, ABI, and symbol checks only — never hashing files or
running first-use GPU numerical self-probes on the serving/load path.
"""
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import lzma
import os
import sys
import types

import pytest

from opt_core import gates

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(CORE_DIR, "opt_core")
TOOL = os.path.join(CORE_DIR, "tools", "binary_sums.py")


def _tool():
    spec = importlib.util.spec_from_file_location("binary_sums_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture()
def fresh(monkeypatch):
    """Empty caches for the test (the process-wide ones are restored after)."""
    monkeypatch.setattr(gates, "_BINARY_VERDICTS", {})
    monkeypatch.setattr(gates, "_SUMS_READ", {})
    monkeypatch.setattr(gates, "BINARIES_HELD", [])
    return gates


@pytest.fixture()
def fakepkg(tmp_path, monkeypatch, fresh):
    """A stand-in package directory for synthetic tests."""
    pkg = tmp_path / "pkg"
    (pkg / "a" / "b").mkdir(parents=True)
    monkeypatch.setattr(gates, "_PACKAGE_DIR", str(pkg))
    return pkg


# ----------------------------------------------------------------------------------------------------------- offline SHA256SUMS tooling

def test_every_shipped_binary_is_a_sha256sums_line_with_its_digest(fresh):
    tool = _tool()
    bins = tool.binaries(PKG)
    assert len(bins) > 0
    assert tool.check(PKG) == [], "\n".join(tool.check(PKG))
    refused = [(os.path.relpath(p, PKG), gates.binary_refusal(p)) for p in bins if gates.binary_refusal(p) is not None]
    assert refused == []
    for p in bins:
        sums_path, key = gates.binary_sums_for(p)
        assert sums_path.startswith(PKG + os.sep) and "/" + key == "/" + os.path.relpath(p, os.path.dirname(sums_path)).replace(os.sep, "/")


def test_every_sha256sums_beside_a_binary_is_in_sha256sum_format():
    tool = _tool()
    for s in tool.sums_files(PKG):
        for n, line in enumerate(open(s, encoding="utf-8"), 1):
            if not line.strip() or line.startswith("#"):
                continue
            digest, rel = line.rstrip("\n").split(None, 1)
            assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), (s, n)
            assert not rel.startswith("/") and ".." not in rel.split("/"), (s, n)


def test_binary_sums_for_resolves_nearest_listing_sha256sums(fakepkg, tmp_path):
    cub = fakepkg / "a" / "b" / "k.cubin"; cub.write_bytes(b"k")
    (fakepkg / "a" / "b" / "SHA256SUMS").write_text("%s  other.so\n" % _sha(b"o"))
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  b/k.cubin\n" % _sha(b"k"))
    assert gates.binary_sums_for(str(cub)) == (str(fakepkg / "a" / "SHA256SUMS"), "b/k.cubin")
    so = fakepkg / "a" / "b" / "u.so"; so.write_bytes(b"u")
    assert gates.binary_sums_for(str(so)) == (str(fakepkg / "a" / "b" / "SHA256SUMS"), None)
    (tmp_path / "SHA256SUMS").write_text("%s  pkg/a/b/z.so\n" % _sha(b"z"))
    z = fakepkg / "a" / "b" / "z.so"; z.write_bytes(b"z")
    (fakepkg / "a" / "b" / "SHA256SUMS").unlink()
    (fakepkg / "a" / "SHA256SUMS").unlink()
    assert gates.binary_sums_for(str(z)) is None


def test_the_tool_writes_sums_only_where_no_sha256sums_above_lists_the_binary(tmp_path):
    tool = _tool()
    pkg = tmp_path / "pkg"; (pkg / "k" / "prebuilt" / "s1").mkdir(parents=True); (pkg / "p" / "v1" / "build").mkdir(parents=True)
    (pkg / "k" / "prebuilt" / "s1" / "e.so").write_bytes(b"e"); (pkg / "k" / "prebuilt" / "s1" / "manifest.json").write_text("{}")
    (pkg / "p" / "v1" / "build" / "u.cubin").write_bytes(b"u")
    (pkg / "p" / "v1" / "SHA256SUMS").write_text("%s  build/u.cubin\n%s  README.md\n" % (_sha(b"u"), "0" * 64))
    assert [os.path.relpath(p, str(pkg)) for p, n in tool.write(str(pkg))] == ["k/prebuilt/s1/SHA256SUMS"]
    assert (pkg / "k" / "prebuilt" / "s1" / "SHA256SUMS").read_text() == "%s  e.so\n" % _sha(b"e")
    assert tool.check(str(pkg)) == []
    (pkg / "p" / "v1" / "build" / "u.cubin").write_bytes(b"u2")
    assert tool.check(str(pkg)) == ["p/v1/build/u.cubin: sha256 differs from p/v1/SHA256SUMS"]


# ----------------------------------------------------------------------------------------------------------- runtime presence-only rule

def test_binary_refusal_checks_presence_only_without_hashing(fakepkg, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gates, "sha256_file", lambda *a, **k: pytest.fail("runtime binary_refusal must not hash files"))
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("runtime binary_refusal must not call hashlib.sha256"))

    so = fakepkg / "a" / "x.so"; so.write_bytes(b"any-bytes")
    assert gates.binary_refusal(str(so)) is None                                # no SHA256SUMS needed at runtime
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  x.so\n" % ("0" * 64))
    assert gates.binary_refusal(str(so)) is None                                # mismatched SHA256SUMS ignored at runtime

    missing_in = fakepkg / "a" / "missing.so"
    assert gates.binary_refusal(str(missing_in)) == "missing: a/missing.so"

    missing_out = tmp_path / "elsewhere" / "missing.so"
    assert gates.binary_refusal(str(missing_out)) == f"missing: {os.path.abspath(str(missing_out))}"

    assert gates.binary_refusal(str(missing_in), data=b"in-hand") is None       # bytes already in memory: accepted without disk/hash
    assert capsys.readouterr().err == ""


# ----------------------------------------------------------------------------------------------------------- runtime loader contracts

def test_ln_ext_loader_checks_presence_only(fakepkg, monkeypatch):
    from opt_core.kernels.ln import ext_loader as L
    monkeypatch.setattr(L, "_sha256", lambda *a, **k: pytest.fail("ln.ext_loader must not hash .so at runtime"))
    so = fakepkg / "a" / "fast_layernorm_ext.so"; so.write_bytes(b"so-bytes")
    (fakepkg / "a" / "SHA256SUMS").write_text("%s  fast_layernorm_ext.so\n" % ("0" * 64))
    m = {"_so": str(so), "_key": "torch9.9.9+cu130-cp311", "so_sha256": "0" * 64}
    L._check_digest(m)                                                          # existing file passes without SHA256 check

    so.unlink()
    with pytest.raises(L.Unavailable) as ei:
        L._check_digest(m)
    assert "no_prebuilt:torch9.9.9+cu130-cp311" in str(ei.value)


def test_flash_transition_checks_presence_and_abi_without_hashing(fresh, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels import transition as TR
    FT = TR.carried_module("flash_sm90a")

    dst = tmp_path / "prebuilt_cell"
    dst.mkdir()
    pytag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    man = {
        "so": "flash_transition_sm90a.so",
        "module_name": "flash_transition_sm90a_stub",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python_tag": pytag,
        "so_sha256": "0" * 64,
        "source_sha256": {"missing.cu": "0" * 64},
    }
    (dst / "manifest.json").write_text(json.dumps(man))
    (dst / man["so"]).write_bytes(b"stub-so")
    (dst / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, man["so"]))

    monkeypatch.setattr(FT, "prebuilt_dir", lambda: str(dst))
    monkeypatch.setattr(FT, "_EXT", None)
    monkeypatch.setattr(FT, "_sha", lambda *a, **k: pytest.fail("flash_sm90a.load must not hash files at runtime"))

    # Sealed presence helper ignores mismatched SHA256SUMS and flags absence
    assert TR._sealed_prebuilt_refusal(FT) is None

    # Direct load via stubbed extension loader checks symbols without hashing
    stub_mod = types.ModuleType(man["module_name"])
    for sym in ("flash_transition", "silu_table", "smem_bytes", "hidden_chunk"):
        setattr(stub_mod, sym, lambda *a, **k: None)

    class _FakeLoader:
        def __init__(self, name, path):
            self.name = name
            self.path = path
        def create_module(self, spec):
            return stub_mod
        def exec_module(self, module):
            pass

    monkeypatch.setattr(importlib.machinery, "ExtensionFileLoader", _FakeLoader)
    loaded = FT.load()
    assert loaded is stub_mod

    # Missing symbol refused
    monkeypatch.setattr(FT, "_EXT", None)
    delattr(stub_mod, "flash_transition")
    with pytest.raises(RuntimeError, match="lacks symbol flash_transition"):
        FT.load()

    # ABI mismatch (python tag / torch) refused
    monkeypatch.setattr(FT, "_EXT", None)
    man_bad_py = dict(man, python_tag="cp999")
    (dst / "manifest.json").write_text(json.dumps(man_bad_py))
    with pytest.raises(RuntimeError, match="prebuilt python tag"):
        FT.load()

    man_bad_torch = dict(man, torch="0.0.0+fake")
    (dst / "manifest.json").write_text(json.dumps(man_bad_torch))
    with pytest.raises(RuntimeError, match="prebuilt was built for torch"):
        FT.load()

    # Missing .so refused by both sealed check and load()
    (dst / "manifest.json").write_text(json.dumps(man))
    (dst / man["so"]).unlink()
    assert TR._sealed_prebuilt_refusal(FT) == "prebuilt flash_transition_sm90a.so missing"
    with pytest.raises(RuntimeError, match="missing"):
        FT.load()


def test_cuda_sm90a_checks_presence_and_abi_without_hashing_or_loadcheck(fresh, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels.triattn import cuda_sm90a as C

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(C, "_EXT", None, raising=False)
    monkeypatch.setattr(C, "_INSTALL_REPORT", {})
    monkeypatch.setattr(C, "_sha", lambda *a, **k: pytest.fail("cuda_sm90a.install must not hash .so at runtime"))
    monkeypatch.setattr(C, "source_sha256", lambda *a, **k: pytest.fail("cuda_sm90a.install must not hash csrc at runtime"))
    monkeypatch.setattr(C, "_loadcheck", lambda *a, **k: pytest.fail("cuda_sm90a.install must not run GPU loadcheck at runtime"))

    root = tmp_path / "root"
    dst = root / os.path.basename(C.prebuilt_dir(str(root)))
    dst.mkdir(parents=True)
    man = {
        "so": "triattn_cuda_sm90a.so",
        "module_name": "triattn_cuda_sm90a_stub",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "so_sha256": "0" * 64,
        "source_sha256": "0" * 64,
    }
    (dst / "manifest.json").write_text(json.dumps(man))
    (dst / "SHA256SUMS").write_text("%s  %s\n" % ("0" * 64, man["so"]))

    # Missing .so refused
    with pytest.raises(C.Refused, match="so_exists.*false"):
        C.install(prebuilt_root=str(root))

    # Present .so with wrong torch version refused
    (dst / man["so"]).write_bytes(b"stub-so")
    (dst / "manifest.json").write_text(json.dumps(dict(man, torch="0.0.0+fake")))
    with pytest.raises(C.Refused, match='"torch": false'):
        C.install(prebuilt_root=str(root))

    # Matching ABI loads directly without SHA256 or GPU loadcheck
    (dst / "manifest.json").write_text(json.dumps(man))
    stub_mod = types.SimpleNamespace(
        fwd=lambda *a, **k: None,
        stage_bias=lambda *a, **k: None,
        stage_mask=lambda *a, **k: None,
        fix_elems=lambda *a, **k: 16,
        smem_bytes=lambda *a, **k: 0,
    )
    monkeypatch.setattr(C, "_load_so", lambda so, name: stub_mod)
    monkeypatch.setattr(C, "_FIX", {})
    monkeypatch.setattr(C, "_fix_buffer", lambda dev, n: C._FIX.__setitem__(dev, torch.zeros(n, dtype=torch.int32)))

    rep = C.install(prebuilt_root=str(root))
    assert rep["route"] == "prebuilt"
    assert C._EXT is stub_mod


def test_apb_dit_exact_loads_without_hashing_or_loadcheck(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels.apb import dit_exact as D

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    monkeypatch.setattr(D, "_MOD", None)
    monkeypatch.setattr(D, "STATS", dict(D.STATS, loadcheck="stale"))
    monkeypatch.setattr(D, "_sha_file", lambda *a, **k: pytest.fail("dit_exact must not hash .so at runtime"))
    monkeypatch.setattr(D, "source_sha256", lambda *a, **k: pytest.fail("dit_exact must not hash csrc at runtime"))
    monkeypatch.setattr(D, "run_loadcheck", lambda *a, **k: pytest.fail("dit_exact must not run GPU loadcheck at runtime"))

    key = "stub_stack"
    pdir = tmp_path / key
    pdir.mkdir()
    monkeypatch.setattr(D, "prebuilt_dir", lambda k=None: str(pdir))
    man = {
        "so": "dit_attn_exact.so",
        "module_name": "dit_attn_exact_stub",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch": ["sm90"],
        "kernel_version": "1.0.0",
        "so_sha256": "0" * 64,
        "source_sha256": "0" * 64,
    }
    (pdir / "manifest.json").write_text(json.dumps(man))

    # Missing .so refused
    with pytest.raises(RuntimeError, match="missing"):
        D.load_prebuilt(key)

    # Torch/CUDA ABI mismatch refused
    (pdir / man["so"]).write_bytes(b"stub-so")
    (pdir / "manifest.json").write_text(json.dumps(dict(man, torch="0.0.0+fake")))
    with pytest.raises(RuntimeError, match="prebuilt built for torch"):
        D.load_prebuilt(key)

    # Valid manifest + present .so loads directly without hashing or loadcheck
    (pdir / "manifest.json").write_text(json.dumps(man))
    inited = []
    stub_mod = types.ModuleType(man["module_name"])
    stub_mod.version = "1.0.0"
    stub_mod.init = lambda: inited.append(True)

    class _FakeLoader:
        def __init__(self, name, path):
            self.name = name
            self.path = path
        def create_module(self, spec):
            return stub_mod
        def exec_module(self, module):
            pass

    monkeypatch.setattr(importlib.machinery, "ExtensionFileLoader", _FakeLoader)
    mod = D.load_prebuilt(key)
    assert mod is stub_mod
    assert inited == [True]
    assert D.STATS["loadcheck"] is None


def test_triattn_exact_cubin_bytes_checks_size_without_sha256(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.kernels.triattn_exact import _prebuilt as P
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        monkeypatch.setattr(torch.compiler, "disable", lambda fn: fn)
    from opt_core.kernels.triattn_exact._prebuilt import cuda_mma as CM

    payload = b"synthetic-cubin-bytes"
    blob = tmp_path / "k.cubin.xz"
    blob.write_bytes(lzma.compress(payload))

    monkeypatch.setattr(P, "MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(P, "_blob_cache", {})
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("cubin_bytes must not call hashlib.sha256 at runtime"))

    build = {"blob": "k.cubin.xz", "cubin_sha256": "0" * 64, "cubin_bytes": len(payload), "route": "test"}
    assert P.cubin_bytes(build) == payload

    # Wrong byte length still refused
    monkeypatch.setattr(P, "_blob_cache", {})
    bad_build = dict(build, cubin_bytes=len(payload) + 1)
    with pytest.raises(P.Refused, match="size mismatch"):
        P.cubin_bytes(bad_build)

    # cuda_mma.resolve does not call tree_fingerprint
    monkeypatch.setattr(CM, "_loaded", {})
    monkeypatch.setattr(CM, "tree_fingerprint", lambda *a, **k: pytest.fail("cuda_mma.resolve must not hash source tree at runtime"))
    monkeypatch.setattr(CM.drv, "backend", lambda: types.SimpleNamespace(driver_version=lambda: 12090))
    monkeypatch.setattr(CM.drv.Device, "get", staticmethod(lambda idx: types.SimpleNamespace(cc=(9, 0))))
    monkeypatch.setattr(CM, "select_build", lambda *a, **k: (None, "stubbed"))
    with pytest.raises(CM.Refused, match="stubbed"):
        CM.resolve(types.SimpleNamespace(index=0))


def test_trimul_tx_sm90a_loads_without_hashing_or_loadcheck(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from opt_core.kernels.trimul.tx_sm90a import ops as O

    monkeypatch.setattr(O, "_EXT", None)
    monkeypatch.setattr(O, "_STATE", {"loaded": None, "calls": 0})
    monkeypatch.setattr(O, "HERE", str(tmp_path))
    monkeypatch.setattr(O, "_sha256", lambda *a, **k: pytest.fail("tx_sm90a.ops.load must not hash files at runtime"))
    monkeypatch.setattr(O, "_loadcheck", lambda *a, **k: pytest.fail("tx_sm90a.ops.load must not run GPU loadcheck at runtime"))

    pre = tmp_path / "prebuilt"
    pre.mkdir()
    tag = O.abi_tag()
    man = {
        "binaries": {
            tag: {
                "file": "tx_sm90a.so",
                "module": "tx_sm90a_stub",
                "sha256": "a" * 64,
                "toolchain": "synthetic",
            }
        },
        "sources": {"csrc/missing.cu": "b" * 64},
    }
    (pre / "manifest.json").write_text(json.dumps(man))

    # Missing .so refused
    with pytest.raises(O.TrimulTxUnavailable, match="absent"):
        O.load()

    # ABI mismatch refused
    (pre / "tx_sm90a.so").write_bytes(b"stub-so")
    (pre / "manifest.json").write_text(json.dumps({"binaries": {"other_abi": man["binaries"][tag]}}))
    with pytest.raises(O.TrimulTxUnavailable, match="no prebuilt binary for"):
        O.load()

    # Present .so + matching ABI loads directly without SHA256 or loadcheck
    (pre / "manifest.json").write_text(json.dumps(man))
    stub_mod = types.ModuleType("tx_sm90a_stub")

    class _FakeLoader:
        def create_module(self, spec):
            return stub_mod
        def exec_module(self, module):
            pass

    fake_spec = importlib.machinery.ModuleSpec("tx_sm90a_stub", _FakeLoader())
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda name, path: fake_spec)
    assert O.load() is stub_mod
    assert O._STATE["loaded"]["tag"] == tag


def test_exactln_prebuilt_index_checks_opts_without_source_hashing(tmp_path, monkeypatch):
    from opt_core.kernels.ln import exactln as X

    pre = tmp_path / "prebuilt"
    pre.mkdir()
    opts = [o.decode() for o in X._opts(None)]
    idx = {"source_sha256": "0" * 64, "opts": opts, "bundles": {"sm90": {"file": "missing.cubin", "kernels": {"k": "k"}}}}
    (pre / "index.json").write_text(json.dumps(idx))

    monkeypatch.setattr(X, "PREBUILT_DIR", str(pre))
    monkeypatch.setattr(X, "_STATE", dict(X._STATE, prebuilt_index=None, prebuilt_off=None, prebuilt_modules={}))
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("exactln._prebuilt_index must not hash source at runtime"))

    got = X._prebuilt_index()
    assert got is not None and got["opts"] == opts

    # Missing bundle file returns None cleanly
    monkeypatch.setattr(X, "_cu", lambda: object())
    assert X._prebuilt_function("k", "90") is None

    # Mismatched compile options still rejected
    (pre / "index.json").write_text(json.dumps(dict(idx, opts=["--different-opt"])))
    monkeypatch.setattr(X, "_STATE", dict(X._STATE, prebuilt_index=None, prebuilt_off=None, prebuilt_modules={}))
    assert X._prebuilt_index() is None
    assert X._STATE["prebuilt_off"] == "built with other compile options"


def test_triattn_native_runtime_defaults_skip_digests_and_byte_gate(tmp_path, monkeypatch):
    from opt_core.kernels.triattn import triattn_native as W

    assert inspect.signature(W.install).parameters["check"].default is False

    # Structural presence helper passes for shipped stack without hashing
    key = sorted(W.stacks_built())[0]
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **k: pytest.fail("_check_prebuilt_present must not hash .so"))
    rep = W._check_prebuilt_present(key)
    assert rep

    # Missing .so refused by name
    monkeypatch.setattr(W, "payload_dir", lambda: str(tmp_path))
    monkeypatch.setattr(W, "stacks_built", lambda: {key: ["triattn_sm90_ext"]})
    with pytest.raises(W.Unavailable) as ei:
        W._check_prebuilt_present(key)
    assert ei.value.kind.startswith("no_prebuilt:")


def test_trimul_native_runtime_defaults_skip_digests_and_byte_gate():
    from opt_core.kernels import trimul as T
    from opt_core.kernels.trimul import native as N

    assert inspect.signature(N.install).parameters["gate"].default is False
    assert inspect.signature(N.forward).parameters["gate"].default is False
    assert inspect.signature(T.native_install).parameters["gate"].default is False
