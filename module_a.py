import numpy as np
import pandas as pd
import random

class FGOGachaSimulator:
    def __init__(self, ssr_rate=0.01, rate_up_ratio=0.7, pity_threshold=330, one_time_pity=True):
        self.ssr_rate = ssr_rate
        self.rate_up_ratio = rate_up_ratio
        self.pity_threshold = pity_threshold
        self.one_time_pity = one_time_pity

    def simulate_roll(self, current_pity_count, pity_exhausted):
        # Check for macro pity
        if (current_pity_count >= self.pity_threshold - 1) and not pity_exhausted:
            return {
                'outcome': 'SSR_RateUp', 
                'rarity': 5, 
                'is_rate_up': 1, 
                'pity_trigger': 1,
                'reset_pity': True
            }
        
        rng = random.random()

        # Check for 5-star
        if rng < self.ssr_rate:
            rate_up_rng = random.random()
            
            if rate_up_rng < self.rate_up_ratio:
                # Rate-up
                return {
                    'outcome': 'SSR_RateUp', 
                    'rarity': 5, 
                    'is_rate_up': 1, 
                    'pity_trigger': 0,
                    'reset_pity': True
                }
            else:
                # Non rate-up 5 star ("spook")
                return {
                    'outcome': 'SSR_Spook', 
                    'rarity': 5, 
                    'is_rate_up': 0, 
                    'pity_trigger': 0,
                    'reset_pity': False 
                }
        
        # Check for 4-star
        elif rng < (self.ssr_rate + 0.03):
             return {
                'outcome': 'SR', 
                'rarity': 4, 
                'is_rate_up': 0, 
                'pity_trigger': 0,
                'reset_pity': False
            }

        # Must be 3-star
        else:
            return {
                'outcome': '3_Star_or_CE', 
                'rarity': 3, 
                'is_rate_up': 0, 
                'pity_trigger': 0,
                'reset_pity': False
            }

    def generate_dataset(self, num_users=1000, target_copies=5):
        all_rolls = []

        for user_id in range(num_users):
            copies_obtained = 0
            pity_counter = 0
            pity_exhausted = False
            total_rolls = 0

            while copies_obtained < target_copies:
                result = self.simulate_roll(pity_counter, pity_exhausted)
                
                # Data before a pull
                roll_data = {
                    'user_id': user_id,
                    'roll_number': total_rolls + 1,
                    'pity_counter_value': pity_counter,
                    'pity_active_state': 1 if (pity_counter >= self.pity_threshold - 1 and not pity_exhausted) else 0,
                    'pity_exhausted': 1 if pity_exhausted else 0,
                    'outcome': result['outcome'],
                    'is_success': result['is_rate_up'],
                    'triggered_guarantee': result['pity_trigger']
                }
                all_rolls.append(roll_data)

                total_rolls += 1
                
                if result['reset_pity']:
                    pity_counter = 0
                    if self.one_time_pity:
                        pity_exhausted = True
                else:
                    if not pity_exhausted:
                        pity_counter += 1
                
                if result['is_rate_up']:
                    copies_obtained += 1
                    
        return pd.DataFrame(all_rolls)

# Initialize with standard FGO rates
sim = FGOGachaSimulator(ssr_rate=0.01, pity_threshold=330, one_time_pity=True)

print("Starting creation of synthetic data")
df = sim.generate_dataset(num_users=500, target_copies=5)

# Save to CSV
df.to_csv('fgo_synthetic_data.csv', index=False)

print("\nDataset preview:")
print(df.head())

print("\nSuccess Rate by Pity State:")
# This should show ~0.008 for normal state, 1.0 for pity state
print(df.groupby('pity_active_state')['is_success'].mean())