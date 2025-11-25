import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def run_module_b():
    df = pd.read_csv('fgo_synthetic_data.csv')

    # Create DAG
    model = DiscreteBayesianNetwork([
        ('pity_active_state', 'is_success')
    ])

    print(f"Nodes: {model.nodes()}")
    print(f"Edges: {model.edges()}")

    # MLE Learning
    model.fit(df, estimator=MaximumLikelihoodEstimator)

    # Displaying CPTs
    for cpd in model.get_cpds():
        print(f"\nLearned CPT for {cpd.variable}:")
        print(cpd)

    # Variable Elimination to compute exact probabilities
    infer = VariableElimination(model)

    # What is the probability of success (is_success=1) given pity is inactive (0)?
    q1 = infer.query(variables=['is_success'], evidence={'pity_active_state': 0})
    print("What is the probability of success (is_success=1) given pity is inactive (0)?")
    print(q1)

    # What is the probability of success given Pity is active (1)?
    q2 = infer.query(variables=['is_success'], evidence={'pity_active_state': 1})
    print("\nWhat is the probability of success given Pity is active (1)?")
    print(q2)
    
    # Compare learned value vs. actual value
    cpd_success = model.get_cpds('is_success')
    
    # (variable_cardinality, evidence_cardinality)
    learned_rate = cpd_success.values[1, 0] 
    
    print(f"Learned Rate (No Pity): {learned_rate:.5f}")
    print(f"Theoretical Rate:       0.00700")
    print(f"Error/Deviation:        {abs(learned_rate - 0.007):.5f}")

if __name__ == "__main__":
    run_module_b()