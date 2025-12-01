import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

print("Generating Synthetic Data & Learning BN")

# Configuration
PROBS = {'3_CE': 0.40, '3_S': 0.40, '4_CE': 0.12, '4_S': 0.03, '5_CE': 0.04, '5_S': 0.01}
card_types = list(PROBS.keys())
weights = list(PROBS.values())

# Handles specific slot targets
def get_folded_weights(target_type):
    new_probs = PROBS.copy()
    invalid_prob_sum = 0.0
    
    # Define what a valid card for this slot is
    if target_type == 'GOLD':
        target_card = '4_CE' # Cheapest 4*
        def is_valid(c): return '4' in c or '5' in c
    elif target_type == 'SERVANT':
        target_card = '3_S'  # Cheapest servant
        def is_valid(c): return 'S' in c
    else:
        return list(PROBS.values())

    # Count up invalid probabilities
    for card, prob in PROBS.items():
        if not is_valid(card):
            invalid_prob_sum += prob
            new_probs[card] = 0.0

    # Fold invalid probabilities into the target
    new_probs[target_card] += invalid_prob_sum
    
    return list(new_probs.values())

# Generate Data
data_rows = []
for _ in range(100000):
    rolls_9 = np.random.choice(card_types, size=9, p=weights)
    
    has_gold_in_9 = any(('4' in c or '5' in c) for c in rolls_9)
    
    if has_gold_in_9:
        gold_state, w_10 = 'Satisfied', weights
    else:
        gold_state, w_10 = 'Need_Gold', get_folded_weights('GOLD')
        
    roll_10 = np.random.choice(card_types, p=w_10)
    
    all_10 = np.append(rolls_9, roll_10)
    has_servant_in_10 = any(('S' in c) for c in all_10)
    
    if has_servant_in_10:
        servant_state, w_11 = 'Satisfied', weights
    else:
        servant_state, w_11 = 'Need_Servant', get_folded_weights('SERVANT')
        
    roll_11 = np.random.choice(card_types, p=w_11)
    
    data_rows.append([gold_state, roll_10, servant_state, roll_11])

df = pd.DataFrame(data_rows, columns=['Gold_State', 'Roll_10', 'Servant_State', 'Roll_11'])

# Save dataset
output_filename = 'simulation_data_dual.csv'
df.to_csv(output_filename, index=False)

# Learn Model
model = DiscreteBayesianNetwork([
    ('Gold_State', 'Roll_10'),
    ('Servant_State', 'Roll_11')
])
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Create Lookup Table for Simulation
def get_table(node_target, node_parent):
    cpd = model.get_cpds(node_target)
    cpd_values = cpd.values 
    parent_names = cpd.state_names[node_parent]
    # Map state names to their probability rows
    return {state: cpd_values[:, i] for i, state in enumerate(parent_names)}

TABLE_GOLD = get_table('Roll_10', 'Gold_State')
TABLE_SERVANT = get_table('Roll_11', 'Servant_State')
roll_names = model.get_cpds('Roll_10').state_names['Roll_10'] # Same names for both slots

print("Model Learned. Starting Simulation")

# Monte Carlo Simulation
def run_simulation(target_np, use_hard_pity, n_sims=2000):
    results = []
    
    for _ in range(n_sims):
        np_count = 0
        total_rolls = 0
        hard_pity_counter = 0
        
        while np_count < target_np:
            batch_9 = np.random.choice(card_types, size=9, p=weights)
            has_gold = False
            has_servant = False
            
            for card in batch_9:
                total_rolls += 1
                hard_pity_counter += 1
                
                # Check Pity/Success
                is_rate_up = (card == '5_S') and (np.random.random() < 0.8)
                
                if is_rate_up:
                    np_count += 1
                    hard_pity_counter = 0
                elif use_hard_pity and hard_pity_counter >= 330:
                    np_count += 1
                    hard_pity_counter = 0
                
                if np_count >= target_np: break
                
                if '4' in card or '5' in card: has_gold = True
                if 'S' in card: has_servant = True
            
            if np_count >= target_np: break
            
            gold_state = 'Satisfied' if has_gold else 'Need_Gold'
            
            total_rolls += 1
            hard_pity_counter += 1
            
            # Use BN Table for Roll 10
            roll_10 = np.random.choice(roll_names, p=TABLE_GOLD[gold_state])
            
            is_rate_up = (roll_10 == '5_S') and (np.random.random() < 0.8)
            
            if is_rate_up:
                np_count += 1
                hard_pity_counter = 0
            elif use_hard_pity and hard_pity_counter >= 330:
                np_count += 1
                hard_pity_counter = 0
                
            if np_count >= target_np: break

            # Update flags including Roll 10 for the next check
            if '4' in roll_10 or '5' in roll_10: has_gold = True
            if 'S' in roll_10: has_servant = True

            servant_state = 'Satisfied' if has_servant else 'Need_Servant'
            
            total_rolls += 1
            hard_pity_counter += 1
            
            # Use BN Table for Roll 11
            roll_11 = np.random.choice(roll_names, p=TABLE_SERVANT[servant_state])
            
            is_rate_up = (roll_11 == '5_S') and (np.random.random() < 0.8)
            
            if is_rate_up:
                np_count += 1
                hard_pity_counter = 0
            elif use_hard_pity and hard_pity_counter >= 330:
                np_count += 1
                hard_pity_counter = 0
                
        results.append(total_rolls)
    return results

# Run Experiments
print("Simulating NP1 (With Pity)")
np1_pity = run_simulation(target_np=1, use_hard_pity=True)
print("Simulating NP1 (No Pity)")
np1_nopity = run_simulation(target_np=1, use_hard_pity=False)

print("Simulating NP5 (With Pity)")
np5_pity = run_simulation(target_np=5, use_hard_pity=True)
print("Simulating NP5 (No Pity)")
np5_nopity = run_simulation(target_np=5, use_hard_pity=False)

# Visualization
sns.set_style("whitegrid")

# NP1 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np1_nopity, bins=50, alpha=0.5, label='No Hard Pity', color='red', density=True)
plt.hist(np1_pity, bins=50, alpha=0.7, label='Standard Pity (330)', color='blue', density=True)
plt.axvline(x=330, color='green', linestyle='--', label='Pity Threshold (330)')
plt.title('Distribution of Rolls Required for NP1 (Dual Guarantee)')
plt.xlabel('Number of Rolls')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('np1_distribution.png')
plt.show()

# NP5 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np5_nopity, bins=50, alpha=0.5, label='No Hard Pity', color='red', density=True)
plt.hist(np5_pity, bins=50, alpha=0.7, label='Standard Pity', color='blue', density=True)
plt.title('Distribution of Rolls Required for NP5 (Dual Guarantee)')
plt.xlabel('Number of Rolls')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('np5_distribution.png')
plt.show()

# CDF plot
def plot_cdf(data, label, color):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals, label=label, color=color, linewidth=2)

plt.figure(figsize=(12, 6))
plot_cdf(np1_pity, 'NP1 (With Pity)', 'blue')
plot_cdf(np1_nopity, 'NP1 (No Pity)', 'red')
plt.axhline(y=0.5, color='grey', linestyle=':', label='50% Chance')
plt.axhline(y=0.99, color='black', linestyle=':', label='99% Chance')
plt.title('Cumulative Probability of Acquiring NP1')
plt.xlabel('Rolls Spent')
plt.ylabel('Probability of Success')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cdf.png')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
data_to_plot = [np1_pity, np1_nopity, np5_pity, np5_nopity]
labels = ['NP1 (With Pity)', 'NP1 (No Pity)', 'NP5 (With Pity)', 'NP5 (No Pity)']
plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
plt.title('Roll Variation: Pity vs No Pity')
plt.ylabel('Rolls Required')
plt.yscale('log') # I am using log scale since some outliers can be big
plt.grid(True, axis='y', which='major', linestyle='-')
plt.savefig('boxplot.png')
plt.show()