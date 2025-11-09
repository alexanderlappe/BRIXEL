from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Optional

def load_msda() -> Optional[object]:
    """
    Load the MultiScaleDeformableAttention .so located next to this file,
    regardless of how Python was launched (Slurm, different CWDs, etc.).
    Registers the loaded module under both package-relative and top-level
    names for backward compatibility.
    Returns the module object or None if not found.
    """
    ops_dir = Path(__file__).resolve().parent
    # Find the compiled extension regardless of exact python/abi tag
    cand = next(ops_dir.glob("MultiScaleDeformableAttention.cpython-*.so"), None)
    if not cand or not cand.exists():
        return None

    # Choose canonical module names
    # package-qualified (preferred)
    pkg_name = __package__.rsplit('.', 1)[0]  # "...ops"
    canonical_name = f"{pkg_name}.MultiScaleDeformableAttention"
    # Also support the legacy top-level name some code uses
    legacy_name = "MultiScaleDeformableAttention"

    # Avoid reloading if already present
    for name in (canonical_name, legacy_name):
        if name in sys.modules:
            return sys.modules[name]

    spec = importlib.util.spec_from_file_location(canonical_name, str(cand))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Register under both names so any import path resolves to the same object
    sys.modules[canonical_name] = module
    sys.modules[legacy_name] = module
    return module
