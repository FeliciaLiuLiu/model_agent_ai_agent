"""
Model Testing Agent - Comprehensive ML model evaluation toolkit.

Usage:
    from adm_central_utility.model_testing_agent import ModelTestingAgent
    agent = ModelTestingAgent(output_dir='./output')
    results = agent.run(model=model, X=X, y=y)
    agent.generate_report(results)
"""
from .matrices.effectiveness import ModelEffectiveness
from .matrices.efficiency import ModelEfficiency
from .matrices.stability import ModelStability
from .matrices.interpretability import ModelInterpretability
from .runner.main import ModelTestingAgent
from .runner.interactive import InteractiveAgent

__all__ = [
    "ModelTestingAgent", "InteractiveAgent",
    "ModelEffectiveness", "ModelEfficiency", "ModelStability", "ModelInterpretability",
]
__version__ = "1.0.0"
