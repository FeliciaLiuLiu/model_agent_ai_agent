"""ADM Central Utility - Model Testing and EDA"""

try:
    if __package__:
        from . import eda
        from .eda import EDA
    else:
        import eda
        from eda import EDA
except Exception:
    eda = None
    EDA = None

try:
    if __package__:
        from . import model_testing_agent
    else:
        import model_testing_agent
except Exception:
    model_testing_agent = None

try:
    if __package__:
        from .eda import EDASpark
    else:
        from eda import EDASpark
except Exception:
    EDASpark = None

try:
    if __package__:
        from . import model_testing_agent_pyspark
    else:
        import model_testing_agent_pyspark
except Exception:
    model_testing_agent_pyspark = None

__version__ = "1.0.0"
__all__ = [
    "model_testing_agent",
    "EDA",
    "EDASpark",
    "eda",
    "model_testing_agent_pyspark",
]
