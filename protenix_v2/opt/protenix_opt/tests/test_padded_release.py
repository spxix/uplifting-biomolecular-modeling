"""CPU tests for the release of the trunk levers' shared padded tri-attention sets (``ptx_trunk2_levers._PAD_BUFS``).

A long-lived service process saw one new padded set per distinct unaligned token count and kept every set for its whole life, so device memory
grew item by item until the trunk of every later item ran out of memory. ``_padded_bufs`` now keeps one P class live, ``pred_release`` drops the
previous item's sets at the next predict, and a set used inside a CUDA-graph capture is never released. The two functions run here from the
kit's own source on CPU tensors (the module itself needs the device stack to import)."""
import ast
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from protenix_opt import pred_release, stack

SRC = os.path.join(stack.kit_home(), "src", "ptx_trunk2_levers.py")
FUNCS = ("release_padded_bufs", "_padded_bufs")


def _levers(capturing=lambda: False):
    """A namespace holding the kit's two functions over fresh _PAD_BUFS / _STATS, with PAD8 on and a controllable capture state."""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert sorted(d.name for d in defs) == sorted(FUNCS)
    cuda = types.SimpleNamespace(is_available=lambda: True, is_current_stream_capturing=capturing)
    fake_torch = types.SimpleNamespace(zeros=torch.zeros, bfloat16=torch.bfloat16, float32=torch.float32, cuda=cuda)
    ns = {"torch": fake_torch, "os": os, "_PAD_BUFS": {}, "_STATS": {}, "_BLK_PADDED": 8}
    exec(compile(ast.Module(body=defs, type_ignores=[]), SRC, "exec"), ns)
    return ns


def _live(ns):
    return sorted(k[0] for k in ns["_PAD_BUFS"])


def test_a_new_p_class_releases_the_previous_ones():
    ns = _levers()
    for n in (13, 21, 27, 35):                                   # four unaligned sizes -> P = 16, 24, 32, 40
        b = ns["_padded_bufs"](None, n, 2, 4, 8, "cpu", None)
        assert b["P"] == ((n + 7) // 8) * 8 and b["N"] == n
        assert _live(ns) == [b["P"]]                             # before the fix: [16], [16, 24], [16, 24, 32], ...
    assert ns["_STATS"]["blk_att_padded_bufs"] == 4 and ns["_STATS"]["blk_att_padded_releases"] == 3


def test_same_p_class_reuses_its_set_and_rezeroes_on_an_n_change():
    ns = _levers()
    b = ns["_padded_bufs"](None, 17, 2, 4, 8, "cpu", -1e9)
    b["q"].fill_(3.0)
    assert ns["_padded_bufs"](None, 17, 2, 4, 8, "cpu", -1e9) is b and float(b["q"].abs().max()) == 3.0   # same N: no re-zero
    assert ns["_padded_bufs"](None, 19, 2, 4, 8, "cpu", -1e9) is b and float(b["q"].abs().max()) == 0.0   # same P, new N: re-zeroed
    assert float(b["bias"][..., 19:].max()) == -1e9 and float(b["bias"][..., :19].abs().max()) == 0.0
    assert "blk_att_padded_releases" not in ns["_STATS"]


def test_second_head_cell_of_the_same_p_stays_live():
    ns = _levers()
    a = ns["_padded_bufs"](None, 30, 2, 4, 8, "cpu", None)          # pair stack cell
    t = ns["_padded_bufs"](None, 30, 1, 4, 4, "cpu", None)          # template stack cell at the same N
    assert len(ns["_PAD_BUFS"]) == 2 and ns["_padded_bufs"](None, 30, 2, 4, 8, "cpu", None) is a and t is not a


def test_release_drops_everything_and_reports_bytes():
    ns = _levers()
    b = ns["_padded_bufs"](None, 30, 2, 4, 8, "cpu", None)
    assert ns["release_padded_bufs"]() == b["bytes"] > 0 and ns["_PAD_BUFS"] == {}
    assert ns["release_padded_bufs"]() == 0
    fresh = ns["_padded_bufs"](None, 30, 2, 4, 8, "cpu", None)       # rebuilt zero-filled on the next use
    assert fresh is not b and float(fresh["q"].abs().max()) == 0.0


def test_a_set_touched_inside_a_capture_is_never_released():
    state = {"capturing": False}
    ns = _levers(lambda: state["capturing"])
    state["capturing"] = True
    g = ns["_padded_bufs"](None, 13, 2, 4, 8, "cpu", None)           # allocated during capture: a graph holds its addresses
    state["capturing"] = False
    assert g.get("graph") is True and ns["_STATS"]["blk_att_padded_alloc_during_capture"] == 1
    e = ns["_padded_bufs"](None, 21, 2, 4, 8, "cpu", None)           # a new P outside capture keeps the pinned set
    assert _live(ns) == [16, 24]
    assert ns["release_padded_bufs"]() == e["bytes"] and _live(ns) == [16]
    eager = ns["_padded_bufs"](None, 27, 2, 4, 8, "cpu", None)
    assert "graph" not in eager and _live(ns) == [16, 32]
    state["capturing"] = True
    assert ns["_padded_bufs"](None, 27, 2, 4, 8, "cpu", None) is eager   # an eager set later captured is pinned too
    state["capturing"] = False
    assert eager["graph"] is True and ns["release_padded_bufs"]() == 0 and _live(ns) == [16, 32]


class _Levers(types.ModuleType):
    def __init__(self):
        super().__init__("ptx_trunk2_levers")
        self.calls = 0

    def release_padded_bufs(self, keep_P=None):
        self.calls += 1
        return 3 * 2**30


def _wrapped(monkeypatch, outcome):
    monkeypatch.setitem(pred_release._STATE, "last", None)
    monkeypatch.setitem(pred_release._STATE, "released", [])
    lev = _Levers()
    monkeypatch.setitem(sys.modules, "ptx_trunk2_levers", lev)

    def orig(self, data):
        if outcome == "raise":
            raise RuntimeError("CUDA out of memory")
        return {"coordinate": torch.zeros(2, 3)}

    return lev, pred_release.make_wrapper(orig)


def test_pred_release_drops_padded_sets_at_every_predict(monkeypatch, capsys):
    lev, predict = _wrapped(monkeypatch, "ok")
    predict(None, {"sample_name": "a"})
    predict(None, {"sample_name": "b"})
    assert lev.calls == 2
    assert "PRED-RELEASE item=a released_gib=0.00 padded_gib=3.00" in capsys.readouterr().out
    assert pred_release.state()["released_gib"] == 6.0


def test_pred_release_drops_padded_sets_after_an_item_that_raised(monkeypatch):
    lev, predict = _wrapped(monkeypatch, "raise")
    with pytest.raises(RuntimeError):
        predict(None, {"sample_name": "a"})
    with pytest.raises(RuntimeError):
        predict(None, {"sample_name": "b"})                          # no `last` from a: the padded sets still leave
    assert lev.calls == 2 and pred_release._STATE["last"] is None


def test_pred_release_without_the_trunk_levers_is_unchanged(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "ptx_trunk2_levers", raising=False)
    monkeypatch.setitem(pred_release._STATE, "last", None)
    predict = pred_release.make_wrapper(lambda self, data: {"x": 1})
    predict(None, {"sample_name": "a"})
    assert "PRED-RELEASE" not in capsys.readouterr().out
