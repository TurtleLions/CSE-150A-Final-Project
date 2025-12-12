import pandas as pd
import numpy as np

def verify_probabilities():

    # Constants
    P_SSR_BASE = 0.01
    
    n = 9
    p_case1_theo = 0.4 ** n
    p_case2_theo = 0.8 ** n - p_case1_theo
    p_case3_theo = 0.56 ** n - p_case1_theo
    p_case4_theo = 1.0 - (p_case1_theo + p_case2_theo + p_case3_theo)

    renorm_rates = {
        'Case 1': (1/44, 0.0227),
        'Case 2': (0.01, 0.01),
        'Case 3': (1/44, 0.0227),
        'Case 4': (0.01, 0.01)
    }

    try:
        df = pd.read_csv('simulation_data.csv')
    except FileNotFoundError:
        print("Error: simulation_data.csv not found.")
        return

    def is_ssr(card_str):
        return card_str == '5_S'

    df['ssr_10'] = df['Roll_10'].apply(is_ssr)
    df['ssr_11'] = df['Roll_11'].apply(is_ssr)

    # Map data states to text cases
    # Case 1: Need_Gold AND Need_Servant
    # Case 2: Need_Gold AND Satisfied
    # Case 3: Satisfied AND Need_Servant
    # Case 4: Satisfied AND Satisfied
    
    def get_case(row):
        gold = row['Gold_State']
        servant = row['Servant_State']
        if gold == 'Need_Gold' and servant == 'Need_Servant':
            return 'Case 1'
        elif gold == 'Need_Gold' and servant == 'Satisfied':
            return 'Case 2'
        elif gold == 'Satisfied' and servant == 'Need_Servant':
            return 'Case 3'
        else:
            return 'Case 4'

    df['Case'] = df.apply(get_case, axis=1)

    total_sims = len(df)
    results = []

    for case_name in ['Case 1', 'Case 2', 'Case 3', 'Case 4']:
        subset = df[df['Case'] == case_name]
        count = len(subset)
        prob_obs = count / total_sims
        
        rate_10_obs = subset['ssr_10'].mean() if count > 0 else 0
        rate_11_obs = subset['ssr_11'].mean() if count > 0 else 0
        
        if case_name == 'Case 1': prob_theo = p_case1_theo
        elif case_name == 'Case 2': prob_theo = p_case2_theo
        elif case_name == 'Case 3': prob_theo = p_case3_theo
        else: prob_theo = p_case4_theo
        
        rate_10_theo, rate_11_theo = renorm_rates[case_name]

        results.append({
            'Case': case_name,
            'Count': count,
            'Prob (Theory)': f"{prob_theo:.5f}",
            'Prob (Obs)': f"{prob_obs:.5f}",
            'R10 SSR (Renorm)': f"{rate_10_theo:.4f}",
            'R10 SSR (Obs)': f"{rate_10_obs:.4f}",
            'R11 SSR (Renorm)': f"{rate_11_theo:.4f}",
            'R11 SSR (Obs)': f"{rate_11_obs:.4f}"
        })

    res_df = pd.DataFrame(results)
    
    print("Comparison of Theoretical vs Observed Distributions & Rates:")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    verify_probabilities()