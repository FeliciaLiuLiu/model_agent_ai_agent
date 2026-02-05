"""Compatibility package wrapper for adm_central_utility imports.

This repo's top-level modules (e.g., eda, model_testing_agent) live at the
project root. Tests import them via the adm_central_utility package name.
Extend this package's search path to include the repo root so those modules
resolve as subpackages (adm_central_utility.eda, etc.).
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

_repo_root = Path(__file__).resolve().parents[1]
_repo_root_str = str(_repo_root)
if _repo_root_str not in __path__:
    __path__.append(_repo_root_str)

# Optional convenience re-exports
try:
    from .eda import EDA  # type: ignore
except Exception:  # pragma: no cover - optional dependency paths
    EDA = None  # type: ignore

try:
    from .eda import EDASpark  # type: ignore
except Exception:  # pragma: no cover - optional dependency paths
    EDASpark = None  # type: ignore

try:
    from . import model_testing_agent  # type: ignore
except Exception:  # pragma: no cover - optional dependency paths
    model_testing_agent = None  # type: ignore

try:
    from . import model_testing_agent_pyspark  # type: ignore
except Exception:  # pragma: no cover - optional dependency paths
    model_testing_agent_pyspark = None  # type: ignore

__all__ = [
    "EDA",
    "EDASpark",
    "model_testing_agent",
    "model_testing_agent_pyspark",
]
