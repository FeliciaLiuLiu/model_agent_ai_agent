"""EDA module."""
from .runner import EDA

try:
    from .spark_runner import EDASpark
except Exception:
    EDASpark = None

__all__ = ["EDA", "EDASpark"]
