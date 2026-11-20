"""CPU checks of the eager DiT hoist (protenix_opt.eager_hoist): the admitted route runs and the buffers leave with it."""
import sys
from collections import namedtuple
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from protenix_opt import eager_hoist  # noqa: E402

Admission = namedtuple("Admission", "route")


class FakeHoist:
    def __init__(self):
        self.entries = {"eager": {"dit_hoist": {"slots": {"tok.znorm": object(), "tok.bias0": object()}}}}
        self._padbias = {"key": object()}
        self.mode = "record"

    def set_mode(self, mode):
        self.mode = mode

    def release(self):
        self.entries.clear()


@pytest.fixture
def kit(monkeypatch):
    calls = []
    fake = FakeHoist()

    def install(model, sampler):
        sampler.biascache = fake
        return fake

    policy = ModuleType("fpf_clisampler.policy")
    policy.route = "hoist_eager"
    policy.admit = lambda **kw: calls.append(("admit", kw["n_tok"], kw["candidates"])) or Admission(policy.route)
    policy.run_hoist_eager = lambda sampler, fn, **kw: calls.append(("hoist", sampler.biascache is fake)) or fn(**kw)
    policy.run_stock = lambda sampler, fn, **kw: calls.append(("stock",)) or fn(**kw)
    policy.note_measured = lambda adm, peak, **kw: calls.append(("measured", adm.route, kw["released"]))
    policy.words = lambda: "admit=memory"
    package = ModuleType("fpf_clisampler")
    package.policy = policy
    generator = ModuleType("protenix.model.protenix")
    generator.sample_diffusion = lambda **kw: calls.append(("sampler", kw["N_sample"])) or "coordinates"
    for name, module in (("dit_hoist", SimpleNamespace(install=install)), ("fpf_clisampler", package),
                         ("fpf_clisampler.policy", policy), ("protenix.model.protenix", generator)):
        monkeypatch.setitem(sys.modules, name, module)
    model = SimpleNamespace(protenix=generator)
    monkeypatch.setitem(sys.modules, "protenix.model", model)
    monkeypatch.setitem(sys.modules, "protenix", SimpleNamespace(model=model))
    for name, value in (("current_device", lambda: 0), ("synchronize", lambda: None),
                        ("get_device_properties", lambda device: SimpleNamespace(total_memory=96 * 2**30)),
                        ("memory_allocated", lambda: 0), ("max_memory_allocated", lambda: 0)):
        monkeypatch.setattr(torch.cuda, name, value)
    return SimpleNamespace(calls=calls, hoist=fake, policy=policy, generator=generator)


def sample(generator):
    features = {"atom_to_token_idx": torch.zeros(40, dtype=torch.long)}
    return generator.sample_diffusion(s_inputs=torch.zeros(1, 12, 4), input_feature_dict=features, N_sample=5)


@pytest.mark.parametrize("route", ["hoist_eager", "stock"])
def test_the_admitted_route_runs_and_the_buffers_are_released(kit, route):
    kit.policy.route = route
    installed = eager_hoist.install(model=None)
    assert sample(kit.generator) == "coordinates"
    run = ("hoist", True) if route == "hoist_eager" else ("stock",)
    assert kit.calls == [("admit", 12, eager_hoist.ROUTES), run, ("sampler", 5),
                         ("measured", route, "released=2 g2_cleared=1")]
    assert installed.routes[route] == 1
    assert kit.hoist.entries == {} and kit.hoist._padbias == {} and kit.hoist.mode == "off"


def test_release_outside_the_sampler_drops_the_buffers(kit):
    installed = eager_hoist.install(model=None)
    assert installed.release() == "released=2 g2_cleared=1"
    assert kit.hoist.entries == {} and kit.hoist._padbias == {} and kit.hoist.mode == "off"
