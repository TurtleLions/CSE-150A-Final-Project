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

def get_renormalized_weights(missing_gold, missing_servant):
    valid_weights = []
    for card in card_types:
        is_servant = 'S' in card
        is_gold = '4' in card or '5' in card
        if ((not missing_gold) or is_gold) and ((not missing_servant) or is_servant):
            valid_weights.append(PROBS[card])
        else:
            valid_weights.append(0.0)
    total = sum(valid_weights)
    return [w / total for w in valid_weights]

# Generate Data
data_rows = []
for _ in range(1000000):
    rolls_10 = np.random.choice(card_types, size=10, p=weights)
    has_gold = any(('4' in c or '5' in c) for c in rolls_10)
    has_servant = any(('S' in c) for c in rolls_10)
    
    if has_gold and has_servant:
        state, w = 'Satisfied', get_renormalized_weights(False, False)
    elif not has_gold and has_servant:
        state, w = 'Need_Gold', get_renormalized_weights(True, False)
    elif has_gold and not has_servant:
        state, w = 'Need_Servant', get_renormalized_weights(False, True)
    else:
        state, w = 'Need_Both', get_renormalized_weights(True, True)
        
    roll_11 = np.random.choice(card_types, p=w)
    data_rows.append([state, roll_11])

df = pd.DataFrame(data_rows, columns=['Latent_State', 'Roll_11'])

# Save dataset
output_filename = 'simulation_data.csv'
df.to_csv(output_filename, index=False)

# Learn Model
model = DiscreteBayesianNetwork([('Latent_State', 'Roll_11')])
model.fit(df, estimator=MaximumLikelihoodEstimator)
learned_cpd = model.get_cpds('Roll_11')

# Create Lookup Table for Simulation
cpd_values = learned_cpd.values 
state_names = learned_cpd.state_names['Latent_State']
roll_names = learned_cpd.state_names['Roll_11']
LEARNED_TABLE = {state: cpd_values[:, i] for i, state in enumerate(state_names)}

print("Model Learned. Starting Simulation")

# Monte Carlo Simulation
def run_simulation(target_np, use_hard_pity, n_sims=10000):
    results_counts = []
    observation_log = []
    
    for sim_id in range(n_sims):
        np_count = 0
        total_rolls = 0
        hard_pity_counter = 0
        
        while np_count < target_np:
            # First 10 rolls
            batch_10 = np.random.choice(card_types, size=10, p=weights)
            has_gold, has_servant = False, False
            
            for card in batch_10:
                total_rolls += 1
                hard_pity_counter += 1
                
                observation_log.append({
                    'sim_id': sim_id,
                    'global_roll_index': total_rolls,
                    'card': card
                })
                
                is_rate_up = (card == '5_S') and (np.random.random() < 0.8)
                
                if is_rate_up:
                    np_count += 1
                    hard_pity_counter = 0
                elif use_hard_pity and hard_pity_counter >= 330:
                    np_count += 1
                    hard_pity_counter = 0
                
                if '4' in card or '5' in card: has_gold = True
                if 'S' in card: has_servant = True

                if np_count >= target_np: break
            
            if np_count >= target_np: break
            
            # Roll 11
            if has_gold and has_servant: state = 'Satisfied'
            elif not has_gold and has_servant: state = 'Need_Gold'
            elif has_gold and not has_servant: state = 'Need_Servant'
            else: state = 'Need_Both'
            
            total_rolls += 1
            hard_pity_counter += 1
            
            roll_11 = np.random.choice(roll_names, p=LEARNED_TABLE[state])
            
            observation_log.append({
                'sim_id': sim_id,
                'global_roll_index': total_rolls,
                'card': roll_11
            })
            
            is_rate_up = (roll_11 == '5_S') and (np.random.random() < 0.8)
            
            if is_rate_up:
                np_count += 1
                hard_pity_counter = 0
            elif use_hard_pity and hard_pity_counter >= 330:
                np_count += 1
                hard_pity_counter = 0
                
        results_counts.append(total_rolls)
        
    return results_counts, pd.DataFrame(observation_log)

# Run Experiments
print("Simulating NP1 (With Pity)")
np1_pity_counts, np1_pity_log = run_simulation(target_np=1, use_hard_pity=True)
np1_pity_log.to_csv('observations_np1_pity.csv', index=False)

print("Simulating NP1 (No Pity)")
np1_nopity_counts, np1_nopity_log = run_simulation(target_np=1, use_hard_pity=False)
np1_nopity_log.to_csv('observations_np1_nohardpity.csv', index=False)

print("Simulating NP5 (With Pity)")
np5_pity_counts, np5_pity_log = run_simulation(target_np=5, use_hard_pity=True)
np5_pity_log.to_csv('observations_np5_pity.csv', index=False)

print("Simulating NP5 (No Pity)")
np5_nopity_counts, np5_nopity_log = run_simulation(target_np=5, use_hard_pity=False)
np5_nopity_log.to_csv('observations_np5_nohardpity.csv', index=False)

print("All observation logs saved.")

# Visualization
sns.set_style("whitegrid")

# NP1 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np1_nopity_counts, bins=50, alpha=0.5, label='No Hard Pity', color='red', density=True)
plt.hist(np1_pity_counts, bins=50, alpha=0.7, label='Standard Pity (330)', color='blue', density=True)
plt.axvline(x=330, color='green', linestyle='--', label='Pity Threshold (330)')
plt.title('Distribution of Rolls Required for NP1 (1-slot Renormalized)')
plt.xlabel('Number of Rolls')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('np1_distribution.png')
plt.show()

# NP5 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np5_nopity_counts, bins=50, alpha=0.5, label='No Hard Pity', color='red', density=True)
plt.hist(np5_pity_counts, bins=50, alpha=0.7, label='Standard Pity', color='blue', density=True)
plt.title('Distribution of Rolls Required for NP5 (1-slot Renormalized)')
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
plot_cdf(np1_pity_counts, 'NP1 (With Pity)', 'blue')
plot_cdf(np1_nopity_counts, 'NP1 (No Pity)', 'red')
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
data_to_plot = [np1_pity_counts, np1_nopity_counts, np5_pity_counts, np5_nopity_counts]
labels = ['NP1 (With Pity)', 'NP1 (No Pity)', 'NP5 (With Pity)', 'NP5 (No Pity)']
plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
plt.title('Roll Variation: Pity vs No Pity')
plt.ylabel('Rolls Required')
plt.yscale('log') # I am using log scale since some outliers can be big
plt.grid(True, axis='y', which='major', linestyle='-')
plt.savefig('boxplot.png')
plt.show()