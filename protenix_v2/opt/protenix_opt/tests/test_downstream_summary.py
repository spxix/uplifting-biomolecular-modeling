"""Synthetic confidence tensors retain every field of the installed upstream."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ConfigDict = pytest.importorskip("ml_collections").ConfigDict
sc = pytest.importorskip("protenix.model.sample_confidence")


@pytest.mark.parametrize("frameless", [False, True])
def test_upstream_summary_fields_are_preserved(frameless):
    source = Path(__file__).resolve().parents[2] / "forward/flashpairformer/src/protenix_ptx_summary_host.py"
    spec = importlib.util.spec_from_file_location("downstream_summary", source)
    sh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sh)
    cfg = ConfigDict({"loss": {
        "pae": {"min_bin": 0, "max_bin": 32, "no_bins": 64},
        "pde": {"min_bin": 0, "max_bin": 32, "no_bins": 64},
        "plddt": {"min_bin": 0, "max_bin": 1, "no_bins": 50}},
        "metrics": {"clash": {"af3_clash_threshold": 1.1}}})
    g = torch.Generator().manual_seed(17)
    frame = torch.ones(12, dtype=torch.long)
    if frameless:
        frame[6:] = 0
    args = dict(configs=cfg, pae_logits=torch.randn(2, 12, 12, 64, generator=g),
                pde_logits=torch.randn(2, 12, 12, 64, generator=g),
                plddt_logits=torch.randn(2, 24, 50, generator=g),
                contact_probs=torch.rand(12, 12, generator=g),
                token_asym_id=torch.tensor([0] * 6 + [1] * 6), token_has_frame=frame,
                atom_coordinate=torch.randn(2, 24, 3, generator=g),
                atom_to_token_idx=torch.arange(12).repeat_interleave(2),
                atom_is_polymer=torch.ones(24, dtype=torch.long), N_recycle=2)
    expected, _ = sc.compute_full_data_and_summary(**args)
    actual, _ = sh.compute_full_data_and_summary(**args)
    for a, b in zip(actual, expected, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
