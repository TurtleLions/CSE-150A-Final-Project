import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting Simulation")

# Configuration
PROBS = {'3_CE': 0.40, '3_S': 0.40, '4_CE': 0.12, '4_S': 0.03, '5_CE': 0.04, '5_S': 0.01}

# Soft pity config
SOFT_PITY_START = 280
SOFT_PITY_INC = 0.02
HARD_PITY_CAP = 330

non_target_cards = [c for c in PROBS.keys() if c != '5_S']
non_target_probs = [PROBS[c] for c in non_target_cards]
non_target_sum = sum(non_target_probs)
non_target_weights = [p / non_target_sum for p in non_target_probs]

# Monte Carlo Simulation
def run_simulation(target_np, use_pity, n_sims=10000):
    results_counts = []
    observation_log = []
    
    for sim_id in range(n_sims):
        np_count = 0
        total_rolls = 0
        pity_counter = 0 # Tracks rolls since last 5* S
        
        while np_count < target_np:
            total_rolls += 1
            pity_counter += 1
            
            current_card = ''
            
            # Hard pity check
            if use_pity and pity_counter >= HARD_PITY_CAP:
                np_count += 1
                pity_counter = 0
                current_card = '5_S'
            else:
                # Soft pity calc
                current_prob = PROBS['5_S']
                
                if use_pity and pity_counter >= SOFT_PITY_START:
                    bonus = (pity_counter - SOFT_PITY_START) * SOFT_PITY_INC
                    current_prob = min(1.0, current_prob + bonus)
                
                # Roll
                if np.random.random() < current_prob:
                    np_count += 1
                    pity_counter = 0
                    current_card = '5_S'
                else:
                    # Generate specific non-target card for the log
                    current_card = np.random.choice(non_target_cards, p=non_target_weights)
            
            # Log the observation
            observation_log.append({
                'sim_id': sim_id,
                'global_roll_index': total_rolls,
                'card': current_card
            })
            
            if np_count >= target_np: break
                
        results_counts.append(total_rolls)
    return results_counts, pd.DataFrame(observation_log)

# Run Experiments
print("Simulating NP1 (With Soft Pity)")
np1_pity_counts, np1_pity_log = run_simulation(target_np=1, use_pity=True)
np1_pity_log.to_csv('observations_np1_pity.csv', index=False)

print("Simulating NP1 (No Pity)")
np1_nopity_counts, np1_nopity_log = run_simulation(target_np=1, use_pity=False)
np1_nopity_log.to_csv('observations_np1_nohardpity.csv', index=False)

print("Simulating NP5 (With Soft Pity)")
np5_pity_counts, np5_pity_log = run_simulation(target_np=5, use_pity=True)
np5_pity_log.to_csv('observations_np5_pity.csv', index=False)

print("Simulating NP5 (No Pity)")
np5_nopity_counts, np5_nopity_log = run_simulation(target_np=5, use_pity=False)
np5_nopity_log.to_csv('observations_np5_nohardpity.csv', index=False)

print("All observation logs saved.")

# Visualization
sns.set_style("whitegrid")

# NP1 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np1_nopity_counts, bins=50, alpha=0.5, label='No Pity', color='red', density=True)
plt.hist(np1_pity_counts, bins=50, alpha=0.7, label=f'Soft Pity (Soft {SOFT_PITY_START}, Hard {HARD_PITY_CAP})', color='blue', density=True)
plt.axvline(x=SOFT_PITY_START, color='orange', linestyle='--', label=f'Soft Start ({SOFT_PITY_START})')
plt.axvline(x=HARD_PITY_CAP, color='green', linestyle='--', label=f'Hard Cap ({HARD_PITY_CAP})')
plt.title('Distribution of Rolls Required for NP1 (Soft pity)')
plt.xlabel('Number of Rolls')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('np1_distribution.png')
plt.show()

# NP5 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np5_nopity_counts, bins=50, alpha=0.5, label='No Pity', color='red', density=True)
plt.hist(np5_pity_counts, bins=50, alpha=0.7, label='Soft Pity', color='blue', density=True)
plt.title('Distribution of Rolls Required for NP5 (Soft pity)')
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
labels = ['NP1 (Pity)', 'NP1 (No Pity)', 'NP5 (Pity)', 'NP5 (No Pity)']
plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
plt.title('Roll Variation: Pity vs No Pity')
plt.ylabel('Rolls Required')
plt.yscale('log') 
plt.grid(True, axis='y', which='major', linestyle='-')
plt.savefig('boxplot.png')
plt.show()