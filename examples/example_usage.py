"""Example usage of Model Testing Agent."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from adm_central_utility.model_testing_agent import ModelTestingAgent, InteractiveAgent


def create_sample_data(n=2000):
    np.random.seed(42)
    X = pd.DataFrame({
        'amount': np.random.exponential(500, n),
        'count_24h': np.random.poisson(5, n),
        'distance': np.random.exponential(50, n),
        'device_score': np.random.beta(5, 2, n),
        'account_age': np.random.exponential(365, n),
    })
    y = ((X['amount'] > 500) & (X['distance'] > 30) | (np.random.random(n) < 0.05)).astype(int)
    return X, y


def example_non_interactive():
    print("\n" + "="*60 + "\nNON-INTERACTIVE MODE\n" + "="*60)
    X, y = create_sample_data()
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

    agent = ModelTestingAgent(output_dir='./output')
    results = agent.run(model=model, X=X, y=y)
    print(f"\nReport: {agent.generate_report(results)}")


def example_interactive():
    print("\n" + "="*60 + "\nINTERACTIVE MODE\n" + "="*60)
    X, y = create_sample_data()
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

    agent = InteractiveAgent(output_dir='./output_interactive')
    agent.run_interactive(model=model, X=X, y=y, feature_names=list(X.columns))


if __name__ == '__main__':
    example_non_interactive()
    # example_interactive()  # Uncomment for interactive
