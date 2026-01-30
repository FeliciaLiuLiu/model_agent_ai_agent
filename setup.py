from setuptools import setup, find_packages

setup(
    name="adm_central_utility",
    version="1.0.0",
    description="Model Testing Agent - Comprehensive ML model evaluation toolkit",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "reportlab>=3.6.0",
        "joblib>=1.1.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "full": ["shap>=0.40.0", "lime>=0.2.0"],
    },
    entry_points={
        "console_scripts": [
            "model-testing-agent=adm_central_utility.model_testing_agent.runner.cli:main",
            "model-testing-agent-spark=adm_central_utility.model_testing_agent_pyspark.runner.cli:main",
            "eda-agent=adm_central_utility.eda.cli:main",
        ],
    },
)
