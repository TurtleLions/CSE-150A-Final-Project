import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

class PityModelClassifier:
    def __init__(self):
        # Config
        self.PROBS = {'3_CE': 0.40, '3_S': 0.40, '4_CE': 0.12, '4_S': 0.03, '5_CE': 0.04, '5_S': 0.01}
        self.CARD_TYPES = list(self.PROBS.keys())
        
        # Pity constants
        self.SOFT_PITY_START = 280
        self.SOFT_PITY_INC = 0.02
        self.HARD_PITY_CAP = 330
        self.EPSILON = 1e-12

    # Weight helper functions
    def get_folded_weights(self, target_type):
        new_probs = self.PROBS.copy()
        invalid_prob_sum = 0.0
        
        # Validation Logic
        if target_type == 'GOLD':
            target_card = '4_CE'
            def is_valid(c): return '4' in c or '5' in c
        elif target_type == 'SERVANT':
            target_card = '3_S'
            def is_valid(c): return 'S' in c
        elif target_type == 'BOTH': # For 1-Slot Logic
            target_card = '4_S'
            def is_valid(c): return ('S' in c) and ('4' in c or '5' in c)
        else:
            return self.PROBS

        # Folding Logic
        for card, prob in self.PROBS.items():
            if not is_valid(card):
                invalid_prob_sum += prob
                new_probs[card] = 0.0
        new_probs[target_card] += invalid_prob_sum
        return new_probs

    def get_renorm_weights(self, target_type):
        valid_weights = {}
        # Validation Logic
        if target_type == 'GOLD':
            def is_valid(c): return '4' in c or '5' in c
        elif target_type == 'SERVANT':
            def is_valid(c): return 'S' in c
        elif target_type == 'BOTH': # For 1-Slot Logic
            def is_valid(c): return ('S' in c) and ('4' in c or '5' in c)
        else:
            return self.PROBS

        # Renormalization Logic
        total_valid_mass = 0
        for card in self.CARD_TYPES:
            if is_valid(card):
                total_valid_mass += self.PROBS[card]
        
        if total_valid_mass == 0: return self.PROBS # Fallback
        
        for card in self.CARD_TYPES:
            if is_valid(card):
                valid_weights[card] = self.PROBS[card] / total_valid_mass
            else:
                valid_weights[card] = 0.0
        return valid_weights

    # 2-slot
    def calc_ll_2slot(self, sequence, model_type):
        log_likelihood = 0.0
        pity_counter = 0
        current_batch = []
        
        for i, card in enumerate(sequence):
            prob = 0.0
            slot_in_batch = i % 11
            active_probs = self.PROBS

            # Batch Logic
            if slot_in_batch < 9:
                current_batch.append(card)
            elif slot_in_batch == 9: # 4* slot
                has_gold = any(('4' in c or '5' in c) for c in current_batch)
                if not has_gold:
                    active_probs = self.get_folded_weights('GOLD') if model_type == 'Folded' else self.get_renorm_weights('GOLD')
                current_batch.append(card)
            elif slot_in_batch == 10: # Servant slot
                has_servant = any('S' in c for c in current_batch)
                if not has_servant:
                    active_probs = self.get_folded_weights('SERVANT') if model_type == 'Folded' else self.get_renorm_weights('SERVANT')
                current_batch = []

            # Hard Pity Check
            if pity_counter >= 330:
                prob = 1.0 if card == '5_S' else 0.0
            else:
                prob = active_probs.get(card, 0.0)

            if card == '5_S': pity_counter = 0
            else: pity_counter += 1
            
            log_likelihood += np.log(max(prob, self.EPSILON))
            
        return log_likelihood

    # 1-slot
    def calc_ll_1slot(self, sequence, model_type):
        log_likelihood = 0.0
        pity_counter = 0
        current_batch = []
        
        for i, card in enumerate(sequence):
            prob = 0.0
            slot_in_batch = i % 11
            active_probs = self.PROBS

            # Batch Logic
            if slot_in_batch < 10:
                current_batch.append(card)
            elif slot_in_batch == 10:
                has_gold = any(('4' in c or '5' in c) for c in current_batch)
                has_servant = any('S' in c for c in current_batch)
                
                target = None
                if not has_gold and not has_servant: target = 'BOTH' 
                elif not has_gold: target = 'GOLD'
                elif not has_servant: target = 'SERVANT'
                
                if target:
                    active_probs = self.get_folded_weights(target) if model_type == 'Folded' else self.get_renorm_weights(target)
                
                current_batch = []

            # Hard Pity Check
            if pity_counter >= 330:
                prob = 1.0 if card == '5_S' else 0.0
            else:
                prob = active_probs.get(card, 0.0)

            if card == '5_S': pity_counter = 0
            else: pity_counter += 1
            
            log_likelihood += np.log(max(prob, self.EPSILON))
            
        return log_likelihood

    # Soft pity
    def calc_ll_soft(self, sequence):
        log_likelihood = 0.0
        pity_counter = 0
        
        for card in sequence:
            prob = 0.0
            if pity_counter >= self.HARD_PITY_CAP:
                prob = 1.0 if card == '5_S' else 0.0
            else:
                base_prob_5s = self.PROBS['5_S']
                if pity_counter >= self.SOFT_PITY_START:
                    bonus = (pity_counter - self.SOFT_PITY_START) * self.SOFT_PITY_INC
                    current_prob_5s = min(1.0, base_prob_5s + bonus)
                else:
                    current_prob_5s = base_prob_5s
                
                if card == '5_S':
                    prob = current_prob_5s
                else:
                    # Distribute remaining
                    prob_rest_total = 1.0 - current_prob_5s
                    orig_sum = sum(self.PROBS[c] for c in self.PROBS if c != '5_S')
                    prob = prob_rest_total * (self.PROBS[card] / orig_sum)

            if card == '5_S': pity_counter = 0
            else: pity_counter += 1
            
            log_likelihood += np.log(max(prob, self.EPSILON))
            
        return log_likelihood
    
    def predict_sequence(self, sequence):
        scores = {
            '2-slot Folded': self.calc_ll_2slot(sequence, 'Folded'),
            '2-slot Renormalized': self.calc_ll_2slot(sequence, 'Renormalized'),
            '1-slot Folded': self.calc_ll_1slot(sequence, 'Folded'),
            '1-slot Renormalized': self.calc_ll_1slot(sequence, 'Renormalized'),
            'Soft Pity': self.calc_ll_soft(sequence)
        }
        
        best_model = max(scores, key=scores.get)
        return best_model, scores

class BatchEvaluator:
    def __init__(self, classifier):
        self.classifier = classifier
        
    def evaluate_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"Skipping {filepath}: File not found.")
            print(f"  (Looked at: {os.path.abspath(filepath)})")
            return None, {}

        norm_path = filepath.replace('\\', '/')
        true_label = "Unknown"
        
        if "softpity" in norm_path:
             true_label = "Soft Pity"
        else:
            # Slot check
            is_2slot = "2-slot" in norm_path
            is_1slot = "1-slot" in norm_path
            
            # Logic check
            is_folded = "Folded" in norm_path
            is_renorm = "Renormalized" in norm_path
            
            if is_2slot and is_folded: true_label = "2-slot Folded"
            elif is_2slot and is_renorm: true_label = "2-slot Renormalized"
            elif is_1slot and is_folded: true_label = "1-slot Folded"
            elif is_1slot and is_renorm: true_label = "1-slot Renormalized"

        print("===================================================")
        print(f"True Label: {true_label}")
        
        df = pd.read_csv(filepath)

        # Group by Simulation ID
        all_sim_ids = df['sim_id'].unique()
        
        # 80/20 Split
        np.random.shuffle(all_sim_ids)
        split_idx = int(len(all_sim_ids) * 0.8)
        train_ids = all_sim_ids[:split_idx]
        test_ids = all_sim_ids[split_idx:]
        
        print(f"Total Sims: {len(all_sim_ids)} | Train: {len(train_ids)} | Test: {len(test_ids)}")
        print(f"Performing classification on Test Set (N={len(test_ids)})")
        
        # Filter DF for test set only
        test_df = df[df['sim_id'].isin(test_ids)]
        sim_groups = test_df.groupby('sim_id')
        
        results = {
            '2-slot Folded': 0, '2-slot Renormalized': 0,
            '1-slot Folded': 0, '1-slot Renormalized': 0,
            'Soft Pity': 0
        }
        correct = 0
        total = 0
        
        for sim_id, group in sim_groups:
            sequence = group['card'].tolist()
            prediction, scores = self.classifier.predict_sequence(sequence)
            
            results[prediction] += 1
            total += 1
            if prediction == true_label:
                correct += 1
                
        # Report
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"Test Set Accuracy: {accuracy:.2f}%")
        print("Confusion Breakdown (Predictions on Test Set):")
        for model, count in results.items():
            if count > 0:
                print(f"  Predicted {model}: {count}")
                
        return true_label, results

if __name__ == "__main__":
    classifier = PityModelClassifier()
    evaluator = BatchEvaluator(classifier)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    datasets = [
        os.path.join(base_dir, '2-slot Folded micropity', 'observations_np5_pity.csv'),
        os.path.join(base_dir, '2-slot Renormalized micropity', 'observations_np5_pity.csv'),
        os.path.join(base_dir, '1-slot Folded micropity', 'observations_np5_pity.csv'),
        os.path.join(base_dir, '1-slot Renormalized micropity', 'observations_np5_pity.csv'),
        os.path.join(base_dir, 'softpity', 'observations_np5_pity.csv')
    ]
    
    print("Starting Batch Evaluation with 80/20 Split")
    print(f"Script Location: {base_dir}")
    
    data = []
    
    for ds in datasets:
        true_lbl, res = evaluator.evaluate_file(ds)
        
        row = res.copy()
        row['True Label'] = true_lbl
        data.append(row)

    # Plot
    df = pd.DataFrame(data)
    
    df.set_index('True Label', inplace=True)
    
    labels = [
        '2-slot Folded', 
        '2-slot Renormalized', 
        '1-slot Folded', 
        '1-slot Renormalized', 
        'Soft Pity'
    ]
    
    df = df[labels]
    df = df.reindex(labels)
    
    # Plotting the Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix of Pity Model Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()