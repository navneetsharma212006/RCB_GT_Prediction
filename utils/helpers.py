import os
import pandas as pd
import numpy as np

def generate_dataset(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    
    other_teams = ['CSK', 'MI', 'KKR', 'RR', 'LSG', 'DC', 'PBKS', 'SRH']
    venues = [
        'M. Chinnaswamy Stadium, Bengaluru',
        'Narendra Modi Stadium, Ahmedabad',
        'HPCA Stadium, Dharamshala',
        'Wankhede Stadium, Mumbai',
        'MA Chidambaram Stadium, Chennai',
        'Eden Gardens, Kolkata',
        'Arun Jaitley Stadium, Delhi',
        'Rajiv Gandhi International Stadium, Hyderabad'
    ]
    
    matches = []
    
    h2h_actuals = [
        {
            'date': '2022-04-30', 'team1': 'GT', 'team2': 'RCB', 
            'toss_winner': 'RCB', 'toss_decision': 'bat', 
            'venue': 'Brabourne Stadium, Mumbai', 'winner': 'GT', 
            'win_by_runs': 0, 'win_by_wickets': 6
        },
        {
            'date': '2022-05-19', 'team1': 'RCB', 'team2': 'GT', 
            'toss_winner': 'GT', 'toss_decision': 'bat', 
            'venue': 'Wankhede Stadium, Mumbai', 'winner': 'RCB', 
            'win_by_runs': 0, 'win_by_wickets': 8
        },
        {
            'date': '2023-05-21', 'team1': 'RCB', 'team2': 'GT', 
            'toss_winner': 'GT', 'toss_decision': 'field', 
            'venue': 'M. Chinnaswamy Stadium, Bengaluru', 'winner': 'GT', 
            'win_by_runs': 0, 'win_by_wickets': 6
        },
        {
            'date': '2024-04-28', 'team1': 'GT', 'team2': 'RCB', 
            'toss_winner': 'RCB', 'toss_decision': 'field', 
            'venue': 'Narendra Modi Stadium, Ahmedabad', 'winner': 'RCB', 
            'win_by_runs': 0, 'win_by_wickets': 9
        },
        {
            'date': '2024-05-04', 'team1': 'RCB', 'team2': 'GT', 
            'toss_winner': 'RCB', 'toss_decision': 'field', 
            'venue': 'M. Chinnaswamy Stadium, Bengaluru', 'winner': 'RCB', 
            'win_by_runs': 0, 'win_by_wickets': 4
        }
    ]
    
    current_match_id = 1
    
    for year in [2022, 2023, 2024, 2025, 2026]:
        if year == 2022:
            rcb_win_prob, gt_win_prob = 0.55, 0.70
            num_matches_per_team = 14
        elif year == 2023:
            rcb_win_prob, gt_win_prob = 0.50, 0.75
            num_matches_per_team = 14
        elif year == 2024:
            rcb_win_prob, gt_win_prob = 0.55, 0.45
            num_matches_per_team = 14
        elif year == 2025:
            rcb_win_prob, gt_win_prob = 0.48, 0.65
            num_matches_per_team = 14
        else:
            rcb_win_prob, gt_win_prob = 0.55, 0.75
            num_matches_per_team = 14
            
        start_date = pd.to_datetime(f'{year}-03-26')
        
        for i in range(num_matches_per_team):
            match_date = (start_date + pd.Timedelta(days=i * 5 + np.random.randint(0, 3))).strftime('%Y-%m-%d')
            
            opp1 = np.random.choice(other_teams)
            venue1 = np.random.choice(venues)
            is_home_rcb = 'Bengaluru' in venue1
            eff_rcb_prob = rcb_win_prob + (0.08 if is_home_rcb else -0.05)
            eff_rcb_prob = np.clip(eff_rcb_prob, 0.3, 0.85)
            
            toss_win_rcb = np.random.choice([True, False])
            toss_winner1 = 'RCB' if toss_win_rcb else opp1
            toss_decision1 = np.random.choice(['field', 'bat'], p=[0.75, 0.25])
            
            if toss_decision1 == 'field':
                chasing_team = toss_winner1
            else:
                chasing_team = opp1 if toss_winner1 == 'RCB' else 'RCB'
            
            if chasing_team == 'RCB':
                eff_rcb_prob += 0.08
            else:
                eff_rcb_prob -= 0.08
                
            rcb_wins = np.random.rand() < eff_rcb_prob
            winner1 = 'RCB' if rcb_wins else opp1
            
            win_by_runs1 = np.random.randint(5, 70) if np.random.rand() < 0.4 else 0
            win_by_wickets1 = np.random.randint(3, 9) if win_by_runs1 == 0 else 0
            if win_by_runs1 == 0 and win_by_wickets1 == 0:
                win_by_wickets1 = 5
                
            matches.append({
                'date': match_date, 'team1': 'RCB', 'team2': opp1,
                'toss_winner': toss_winner1, 'toss_decision': toss_decision1,
                'venue': venue1, 'winner': winner1,
                'win_by_runs': win_by_runs1, 'win_by_wickets': win_by_wickets1
            })
            
            opp2 = np.random.choice(other_teams)
            venue2 = np.random.choice(venues)
            is_home_gt = 'Ahmedabad' in venue2
            eff_gt_prob = gt_win_prob + (0.08 if is_home_gt else -0.05)
            eff_gt_prob = np.clip(eff_gt_prob, 0.3, 0.85)
            
            toss_win_gt = np.random.choice([True, False])
            toss_winner2 = 'GT' if toss_win_gt else opp2
            toss_decision2 = np.random.choice(['field', 'bat'], p=[0.75, 0.25])
            
            if toss_decision2 == 'field':
                chasing_team2 = toss_winner2
            else:
                chasing_team2 = opp2 if toss_winner2 == 'GT' else 'GT'
                
            if chasing_team2 == 'GT':
                eff_gt_prob += 0.08
            else:
                eff_gt_prob -= 0.08
                
            gt_wins = np.random.rand() < eff_gt_prob
            winner2 = 'GT' if gt_wins else opp2
            
            win_by_runs2 = np.random.randint(5, 70) if np.random.rand() < 0.4 else 0
            win_by_wickets2 = np.random.randint(3, 9) if win_by_runs2 == 0 else 0
            if win_by_runs2 == 0 and win_by_wickets2 == 0:
                win_by_wickets2 = 5
                
            matches.append({
                'date': match_date, 'team1': 'GT', 'team2': opp2,
                'toss_winner': toss_winner2, 'toss_decision': toss_decision2,
                'venue': venue2, 'winner': winner2,
                'win_by_runs': win_by_runs2, 'win_by_wickets': win_by_wickets2
            })
            
        matches.append({
            'date': f'{year}-04-18' if year != 2026 else '2026-04-18', 
            'team1': 'RCB', 'team2': 'GT',
            'toss_winner': 'GT', 'toss_decision': 'field',
            'venue': 'M. Chinnaswamy Stadium, Bengaluru', 
            'winner': 'GT' if year in [2025, 2026] else 'RCB',
            'win_by_runs': 0, 'win_by_wickets': 7
        })
        matches.append({
            'date': f'{year}-05-10' if year != 2026 else '2026-05-10', 
            'team1': 'GT', 'team2': 'RCB',
            'toss_winner': 'GT', 'toss_decision': 'field',
            'venue': 'Narendra Modi Stadium, Ahmedabad', 
            'winner': 'GT' if year == 2025 else 'RCB',
            'win_by_runs': 30 if year == 2025 else 0, 
            'win_by_wickets': 0 if year == 2025 else 3
        })

    matches.extend(h2h_actuals)
    
    df = pd.DataFrame(matches)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    df = df.drop_duplicates(subset=['date', 'team1', 'team2']).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"[DATASET GENERATOR] Successfully generated {len(df)} matches and saved to '{output_path}'.")
    return df

def print_section(title):
    border = "=" * 80
    print(f"\n{border}")
    print(f" {title.upper()} ".center(80, "="))
    print(f"{border}\n")

def explain_theory(topic):
    explanations = {
        'features_labels': (
            "[THEORY] ML BASICS: FEATURES vs LABELS\n"
            "---------------------------------\n"
            "- FEATURES (X): The inputs or predictors. In our cricket model, these are historical\n"
            "  statistics computed dynamically (e.g., RCB's recent form, GT's momentum, toss success).\n"
            "- LABELS (y): The output we want to predict. Here, it is the 'winner' of the match (RCB or GT).\n"
            "- GOAL: The model learns a mathematical function f(X) = y that maps input features to target labels."
        ),
        'random_forest': (
            "[THEORY] HOW RANDOM FOREST WORKS\n"
            "--------------------------\n"
            "- ENSEMBLE LEARNING: Instead of relying on a single Decision Tree (which easily overfits),\n"
            "  Random Forest builds a collection (an ensemble) of hundreds of trees (300 in our model!).\n"
            "- BAGGING (Bootstrap Aggregating): Each tree is trained on a random subset of the data with replacement.\n"
            "- RANDOM SUBSET OF FEATURES: When splitting nodes, each tree looks at a random subset of features,\n"
            "  ensuring the trees are highly diverse and decorrelated.\n"
            "- DEMOCRATIC VOTING: To make a final prediction, every tree in the forest casts a 'vote'.\n"
            "  The class with the majority votes becomes the predicted winner, and the proportion of votes\n"
            "  defines the 'Win Probability'."
        ),
        'data_leakage': (
            "[THEORY] PREVENTION OF DATA LEAKAGE\n"
            "-----------------------------\n"
            "- WHAT IS LEAKAGE? In sports prediction, calculating recent form or Head-to-Head using the whole\n"
            "  dataset creates leakage. It means a match on '2023-04-10' would use data from a match on '2024-05-04'\n"
            "  to calculate recent form. This is 'looking into the future' and ruins model generalization.\n"
            "- OUR LEAKAGE-FREE ENGINE: For each match played on Date 'D', our pipeline filters the database\n"
            "  to matches that occurred *before* Date 'D'. This ensures the model learns only from real-time historical states."
        ),
        'categorical_encoding': (
            "[THEORY] WHY CATEGORICAL ENCODING IS NEEDED\n"
            "-------------------------------------\n"
            "- MATHEMATICAL INPUTS: Random Forest, like most machine learning models, works purely with numbers\n"
            "  (integers and floats). It cannot directly understand text strings like 'RCB', 'GT', or 'HPCA Stadium'.\n"
            "- LABEL ENCODING: We map categorical strings to unique integers (e.g., 'RCB' -> 0, 'GT' -> 1).\n"
            "- INFERENCE SYNC: We save our LabelEncoders using joblib so we can apply the exact same mapping\n"
            "  on new data during Qualifier 1 predictions."
        )
    }
    
    if topic in explanations:
        border = "-" * 80
        print(f"\n{border}")
        print(explanations[topic])
        print(f"{border}\n")
