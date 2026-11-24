"""binary_sums.py — runtime structural presence checks for the kit's prebuilt kernel binaries.

The prebuilt binaries this kit ships under ``$FPF_HOME/third_party/`` (the stream-correct fast-LayerNorm extension of
``fastln_prebuilt*/`` and the sm_90a tri-attention prologue library of ``protenix_fpf_triatt_procuda/prebuilt/``) each sit beside a
``SHA256SUMS`` in the shared core's format (``<sha256>  <name>``; written/verified offline by the core's ``tools/binary_sums.py``).
At runtime, :func:`binary_refusal` and :func:`manifest_binary_refusal` perform structural file-presence checks only (zero SHA256SUMS
reading or file hashing on the serving path).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from opt_core import gates as _gates

SUMS = _gates.BINARY_SUMS                     # "SHA256SUMS"
_VERDICTS: dict = {}


def binary_refusal(path: str) -> Optional[str]:
    """Runtime presence check for a kit binary at ``path``: returns ``None`` when the file exists, else a missing-file reason
    (no runtime SHA256SUMS reading or file hashing; use ``tools/binary_sums.py`` offline)."""
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        return f"missing: {p}"
    return None


def manifest_binary_refusal(pre_dir: str, key: str = "so") -> Optional[str]:
    """Runtime presence check for the binary a prebuilt directory's ``manifest.json`` names under ``key`` (no SHA256SUMS hashing)."""
    try:
        with open(os.path.join(pre_dir, "manifest.json"), encoding="utf-8") as fh:
            name = json.load(fh).get(key)
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(name, str) or not name:
        return None
    so = os.path.join(pre_dir, name)
    if not os.path.isfile(so):
        return None
    return binary_refusal(so)
