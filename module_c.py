import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LEARNED_RATE = 0.00708
PITY_THRESHOLD = 330
TARGET_COPIES = 5
SIMULATION_RUNS = 10000

class NP5_MonteCarlo_Simulation:
    def __init__(self, rate, pity_limit):
        self.rate = rate
        self.pity_limit = pity_limit

    def simulate_one_user_pity(self):
        # Rolls with pity system
        copies = 0
        total_rolls = 0
        pity_counter = 0
        
        # Pity is one time per banner
        pity_used = False

        while copies < TARGET_COPIES:
            total_rolls += 1
            
            # Pity guarantee
            if not pity_used and pity_counter >= (self.pity_limit - 1):
                copies += 1
                pity_counter = 0
                pity_used = True
                continue

            # Random Roll
            if np.random.random() < self.rate:
                copies += 1
                pity_counter = 0
                # Pity is consumed anyway
                pity_used = True 
            else:
                if not pity_used:
                    pity_counter += 1
        
        return total_rolls

    def simulate_one_user_naive(self):
        # Rolls without pity system
        rolls = sum(np.random.geometric(p=self.rate) for _ in range(TARGET_COPIES))
        return rolls

    def run_simulation(self, n_runs):
        
        pity_results = []
        naive_results = []

        for _ in range(n_runs):
            pity_results.append(self.simulate_one_user_pity())
            naive_results.append(self.simulate_one_user_naive())

        return np.array(pity_results), np.array(naive_results)

def analyze_results(pity_data, naive_data):
    avg_pity = np.mean(pity_data)
    avg_naive = np.mean(naive_data)
    
    p95_pity = np.percentile(pity_data, 95)
    p95_naive = np.percentile(naive_data, 95)

    print("\nResults:")
    print(f"Average Rolls (With Pity):    {avg_pity:.2f}")
    print(f"Average Rolls (Without Pity):      {avg_naive:.2f}")
    print(f"95th Percentile (With Pity):  {p95_pity:.0f}")
    print(f"95th Percentile (Without Pity):    {p95_naive:.0f}")
    print(f"Pity saves approx {avg_naive - avg_pity:.2f} rolls on average")

    plt.figure(figsize=(10, 6))

    plt.hist(naive_data, bins=50, alpha=0.5, label='Naive (Without Pity)', color='red', density=True)
    plt.hist(pity_data, bins=50, alpha=0.7, label='FGO (One Time Pity)', color='blue', density=True)
    
    plt.axvline(avg_pity, color='blue', linestyle='dashed', linewidth=2, label=f'Avg with Pity: {avg_pity:.0f}')
    plt.axvline(avg_naive, color='red', linestyle='dashed', linewidth=2, label=f'Avg without Pity: {avg_naive:.0f}')
    
    plt.title('Distribution of Rolls Needed for NP5')
    plt.xlabel('Total Rolls Needed')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('np5_distribution_analysis.png')
    plt.show()

if __name__ == "__main__":
    sim = NP5_MonteCarlo_Simulation(rate=LEARNED_RATE, pity_limit=PITY_THRESHOLD)
    pity_dist, naive_dist = sim.run_simulation(SIMULATION_RUNS)
    analyze_results(pity_dist, naive_dist)