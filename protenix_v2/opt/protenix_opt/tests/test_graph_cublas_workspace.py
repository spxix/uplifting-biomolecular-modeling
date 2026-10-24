"""cuBLAS workspace vs graph lifetime (infopt_graphs.core.capture_context / clear_cublas_workspaces).

torch 2.13 CUDAGraph::~CUDAGraph() frees the cached cuBLAS workspace of every handle on the destroyed graph's capture stream, and
torch.cuda.Stream() is a 32-stream round-robin pool, so two graphs of the kit regularly share a capture stream.  A graph that baked an
eager-allocated workspace (the warm-up on its capture stream allocates it in the default pool) then replays into memory that
empty_cache has unmapped once any other graph on that stream is dropped: 'unspecified launch failure' / 'illegal memory access'.
The scenario runs in a child process (a CUDA fault is sticky) on a CUDA host; elsewhere it skips by name."""
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "forward", "flashpairformer", "src"))

SCENARIO = textwrap.dedent("""
    import torch
    from infopt_graphs.core import capture_context
    bf = torch.bfloat16
    a = torch.randn(16, 4096, dtype=bf, device="cuda"); w = torch.randn(4096, 512, dtype=bf, device="cuda")
    f = lambda: a @ w
    S = torch.cuda.Stream()
    S.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(S):
        f()                                     # eager warm-up on the capture stream: the workspace lands in the default pool
    torch.cuda.synchronize()
    def cap():
        g = torch.cuda.CUDAGraph()
        with capture_context(g, pool=torch.cuda.graph_pool_handle(), stream=S):
            y = f()
        return g, y
    g_live, y_live = cap()
    g_live.replay(); torch.cuda.synchronize(); ref = y_live.clone()
    g_drop, y_drop = cap()
    del g_drop, y_drop                          # a second graph on the same capture stream is dropped (sampler renew)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    for _ in range(3):
        g_live.replay()
    torch.cuda.synchronize()
    print("EQUAL" if torch.equal(y_live, ref) else "DIFF")
""")


def _cuda():
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")
def test_dropping_a_graph_on_the_same_capture_stream_keeps_the_live_graph_replayable():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    r = subprocess.run([sys.executable, "-c", SCENARIO], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "EQUAL"
