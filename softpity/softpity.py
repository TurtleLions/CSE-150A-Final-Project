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

# Monte Carlo Simulation
def run_simulation(target_np, use_pity, n_sims=2000):
    results = []
    
    for _ in range(n_sims):
        np_count = 0
        total_rolls = 0
        pity_counter = 0 # Tracks rolls since last 5* S
        
        while np_count < target_np:
            total_rolls += 1
            pity_counter += 1
            
            # Hard pity check
            if use_pity and pity_counter >= HARD_PITY_CAP:
                np_count += 1
                pity_counter = 0
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
            
            if np_count >= target_np: break
                
        results.append(total_rolls)
    return results

# Run Experiments
print("Simulating NP1 (With Soft Pity)...")
np1_pity = run_simulation(target_np=1, use_pity=True)
print("Simulating NP1 (No Pity)...")
np1_nopity = run_simulation(target_np=1, use_pity=False)

print("Simulating NP5 (With Soft Pity)...")
np5_pity = run_simulation(target_np=5, use_pity=True)
print("Simulating NP5 (No Pity)...")
np5_nopity = run_simulation(target_np=5, use_pity=False)

# Visualization
sns.set_style("whitegrid")

# NP1 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np1_nopity, bins=50, alpha=0.5, label='No Pity', color='red', density=True)
plt.hist(np1_pity, bins=50, alpha=0.7, label=f'Soft Pity (Soft {SOFT_PITY_START}, Hard {HARD_PITY_CAP})', color='blue', density=True)
plt.axvline(x=SOFT_PITY_START, color='orange', linestyle='--', label=f'Soft Start ({SOFT_PITY_START})')
plt.axvline(x=HARD_PITY_CAP, color='green', linestyle='--', label=f'Hard Cap ({HARD_PITY_CAP})')
plt.title('Distribution of Rolls Required for NP1')
plt.xlabel('Number of Rolls')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('np1_distribution.png')
plt.show()

# NP5 Distribution
plt.figure(figsize=(12, 6))
plt.hist(np5_nopity, bins=50, alpha=0.5, label='No Pity', color='red', density=True)
plt.hist(np5_pity, bins=50, alpha=0.7, label='Soft Pity', color='blue', density=True)
plt.title('Distribution of Rolls Required for NP5')
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
labels = ['NP1 (Pity)', 'NP1 (No Pity)', 'NP5 (Pity)', 'NP5 (No Pity)']
plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
plt.title('Roll Variation: Pity vs No Pity')
plt.ylabel('Rolls Required')
plt.yscale('log') 
plt.grid(True, axis='y', which='major', linestyle='-')
plt.savefig('boxplot.png')
plt.show()