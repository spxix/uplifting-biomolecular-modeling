import pytest

from protenix_opt.modes import layernorm_overrides


def test_exact_torch_preserves_module_normalization():
    overrides = layernorm_overrides("exact", {"LAYERNORM_TYPE": "torch"})
    assert overrides == {"PTX_MK_PF": "0", "PTX_TRIATT_PROCUDA": "0",
                         "PTX_BLOCKFUSE_XL": "0", "PTX_BLK_LN": "stock"}


@pytest.mark.parametrize("mode", ["exact", "fast", "big", "off"])
def test_default_layernorm_keeps_release_modes(mode):
    assert layernorm_overrides(mode, {}) == {}


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_unverified_torch_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="off/exact"):
        layernorm_overrides(mode, {"LAYERNORM_TYPE": "torch"})


def test_caller_cannot_reenable_incompatible_fusion():
    with pytest.raises(ValueError, match="conflicts"):
        layernorm_overrides("exact", {"LAYERNORM_TYPE": "torch", "PTX_MK_PF": "F1"})
