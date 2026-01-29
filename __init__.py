"""ADM Central Utility - Model Testing and EDA"""
from . import eda
from . import model_testing_agent
from .eda import EDA

try:
    from . import model_testing_agent_pyspark
except Exception:
    model_testing_agent_pyspark = None

__version__ = "1.0.0"
__all__ = ["model_testing_agent", "EDA", "eda", "model_testing_agent_pyspark"]
