"""Plans on disk, built once: `compile(fn, cache_dir=...)` returns the plan file of a CasADi
Function (optionally behind a composition) and builds it only when the cache has none.

    from cusadi2 import compile
    plan = compile(fn, cache_dir="/path/.cusadi2", max_live="device", block_size=None)
    plan.path   # <cache_dir>/plans/<hash>.bin

The key is a hash of everything the file depends on: the Function (`fn.serialize()`), the
composition's tables, the resolved `max_live` and `block_size`, the plan format version and the
source of the cusadi2 package (`code_fingerprint`), so an edited cusadi2 checkout does not reuse
plans written by the old code. `max_live="device"` is
resolved to this GPU's slot budget before hashing: a plan is reused only where its window fits.

A build writes `<hash>.bin.tmp.<pid>` and renames it into place, so a reader never sees a partial
plan and two processes building the same plan both end with the same complete file. The cache is
safe to delete at any time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Optional, Union

import casadi as ca
import numpy as np

from cusadi2.compose import Composition
from cusadi2.dag import extract_level_plan
from cusadi2.plan import VERSION, device_max_live, write_plan

_HERE = os.path.dirname(os.path.abspath(__file__))
_code_hash = None


def code_fingerprint() -> bytes:
    """Hash of every Python source of the cusadi2 package (planner, op table, kernel template,
    composition, JAX interpreter): what decides a plan file's bytes."""
    global _code_hash
    if _code_hash is None:
        h = hashlib.sha256()
        for root, dirs, files in os.walk(_HERE):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(f for f in files if f.endswith(".py")):
                path = os.path.join(root, name)
                with open(path, "rb") as fh:
                    h.update(os.path.relpath(path, _HERE).encode() + b"\0" + fh.read())
        _code_hash = h.digest()
    return _code_hash


def feed(h, obj) -> None:
    """Hash `obj` canonically into `h`: dataclasses by field, dicts by sorted key, arrays by
    dtype, shape and bytes, everything else by repr."""
    _feed(h, obj)


def _feed(h, obj) -> None:
    if isinstance(obj, dict):
        h.update(b"M%d" % len(obj))
        for k in sorted(obj, key=repr):
            _feed(h, k)
            _feed(h, obj[k])
    elif dataclasses.is_dataclass(obj):
        h.update(b"D" + type(obj).__name__.encode())
        for f in dataclasses.fields(obj):
            h.update(f.name.encode())
            _feed(h, getattr(obj, f.name))
    elif isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(b"A" + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    elif isinstance(obj, (list, tuple)):
        h.update(b"L%d" % len(obj))
        for v in obj:
            _feed(h, v)
    else:
        h.update(b"S" + repr(obj).encode())


def resolve_max_live(max_live: Union[int, str, None], dtype: str = "double") -> Optional[int]:
    """"device": this GPU's slot budget (`device_max_live`); an int or None as given."""
    if max_live == "device":
        return device_max_live(dtype)
    if max_live is not None and not isinstance(max_live, (int, np.integer)):
        raise ValueError(f"cusadi2: max_live must be an int, None or 'device', got {max_live!r}")
    return None if max_live is None else int(max_live)


@dataclasses.dataclass
class CompiledPlan:
    """A plan file in the cache and the planner's numbers for it."""
    path: str
    key: str
    name: str
    peak_live_slots: int
    n_levels: int
    n_nodes: int
    built: bool          # False: taken from the cache


def plan_key(fn: ca.Function, composition: Optional[Composition], max_live: Optional[int],
             block_size: Optional[int]) -> str:
    h = hashlib.sha256()
    h.update(b"cusadi2-plan\0%d\0" % VERSION)
    h.update(code_fingerprint())
    h.update(fn.serialize().encode())
    _feed(h, composition)
    _feed(h, (max_live, block_size))
    return h.hexdigest()


def compile(fn: ca.Function, *, cache_dir: str, composition: Optional[Composition] = None,
            max_live: Union[int, str, None], block_size: Optional[int]) -> CompiledPlan:
    """The plan of `fn` (behind `composition` if given) from `<cache_dir>/plans/`, built there
    first if missing. `max_live`: slot budget the level plan is windowed to (int), "device" for
    this GPU's, None for no windowing. `block_size`: threads per block recorded in the plan
    (None: the runtime's choice)."""
    max_live = resolve_max_live(max_live)
    key = plan_key(fn, composition, max_live, block_size)
    plans = os.path.join(cache_dir, "plans")
    path, meta_path = os.path.join(plans, key + ".bin"), os.path.join(plans, key + ".json")
    if os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        return CompiledPlan(path=path, key=key, built=False, **meta)
    os.makedirs(plans, exist_ok=True)
    plan = extract_level_plan(fn, max_live=max_live)
    meta = {"name": fn.name(), "peak_live_slots": plan.peak_live_slots, "n_levels": plan.n_levels,
            "n_nodes": plan.n_nodes}
    tmp = f"{path}.tmp.{os.getpid()}"
    write_plan(fn, tmp, block_size=block_size, plan=plan, composition=composition)
    with open(f"{meta_path}.tmp.{os.getpid()}", "w") as fh:
        json.dump(meta, fh)
    os.replace(f"{meta_path}.tmp.{os.getpid()}", meta_path)
    os.replace(tmp, path)
    return CompiledPlan(path=path, key=key, built=True, **meta)
