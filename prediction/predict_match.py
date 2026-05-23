import os
import pandas as pd
import numpy as np
import joblib
from utils.helpers import print_section

def predict_qualifier_1(df, model_dir='model/', venue='HPCA Stadium, Dharamshala'):
    model_path = os.path.join(model_dir, 'random_forest.pkl')
    venue_encoder_path = os.path.join(model_dir, 'venue_encoder.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(venue_encoder_path):
        raise FileNotFoundError("Model or Encoder files not found. Please run training first!")
        
    rf_model = joblib.load(model_path)
    venue_encoder = joblib.load(venue_encoder_path)
    
    t1 = 'RCB'
    t2 = 'GT'
    
    t1_matches = df[(df['team1'] == t1) | (df['team2'] == t1)].sort_values(by='date', ascending=False)
    t2_matches = df[(df['team1'] == t2) | (df['team2'] == t2)].sort_values(by='date', ascending=False)
    
    t1_last_5 = t1_matches.head(5)
    t2_last_5 = t2_matches.head(5)
    rf_1 = sum(t1_last_5['winner'] == t1) / 5.0
    rf_2 = sum(t2_last_5['winner'] == t2) / 5.0
    
    weights_5 = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
    
    def calc_momentum(last_5_df, team):
        components = []
        for _, match in last_5_df.iterrows():
            is_win = match['winner'] == team
            score = 0.0
            if is_win:
                if match['win_by_runs'] > 25 or match['win_by_wickets'] > 5:
                    score = 1.2
                else:
                    score = 1.0
            else:
                team_batted_first = (
                    (match['toss_winner'] == team and match['toss_decision'] == 'bat') or
                    (match['toss_winner'] != team and match['toss_decision'] == 'field')
                )
                is_close = False
                if team_batted_first and match['win_by_wickets'] > 0 and match['win_by_wickets'] <= 3:
                    is_close = True
                elif not team_batted_first and match['win_by_runs'] > 0 and match['win_by_runs'] <= 10:
                    is_close = True
                score = 0.2 if is_close else 0.0
            components.append(score)
        return float(np.sum(np.array(components) * weights_5))
        
    mom_1 = calc_momentum(t1_last_5, t1)
    mom_2 = calc_momentum(t2_last_5, t2)
    
    h2h_matches = df[((df['team1'] == t1) & (df['team2'] == t2)) | ((df['team1'] == t2) & (df['team2'] == t1))]
    t1_h2h_rate = sum(h2h_matches['winner'] == t1) / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
    t2_h2h_rate = 1.0 - t1_h2h_rate
    
    t1_venue_matches = t1_matches[t1_matches['venue'] == venue]
    t2_venue_matches = t2_matches[t2_matches['venue'] == venue]
    ven_1 = sum(t1_venue_matches['winner'] == t1) / len(t1_venue_matches) if len(t1_venue_matches) > 0 else 0.5
    ven_2 = sum(t2_venue_matches['winner'] == t2) / len(t2_venue_matches) if len(t2_venue_matches) > 0 else 0.5
    
    def get_chase_defend_rates(team_df, team):
        chases, defends = [], []
        for _, match in team_df.iterrows():
            bat_second = (
                (match['toss_winner'] == team and match['toss_decision'] == 'field') or
                (match['toss_winner'] != team and match['toss_decision'] == 'bat')
            )
            if bat_second:
                chases.append(match)
            else:
                defends.append(match)
        chase_df = pd.DataFrame(chases)
        defend_df = pd.DataFrame(defends)
        c_rate = sum(chase_df['winner'] == team) / len(chase_df) if len(chase_df) > 0 else 0.5
        d_rate = sum(defend_df['winner'] == team) / len(defend_df) if len(defend_df) > 0 else 0.5
        return c_rate, d_rate
        
    ch_1, def_1 = get_chase_defend_rates(t1_matches, t1)
    ch_2, def_2 = get_chase_defend_rates(t2_matches, t2)
    
    if venue in venue_encoder.classes_:
        venue_encoded = venue_encoder.transform([venue])[0]
    else:
        venue_encoded = len(venue_encoder.classes_)
        
    scenarios = [
        {
            'name': 'Scenario A: RCB wins toss and bowls first (Chasing)',
            'toss_winner': 'RCB', 'toss_decision': 'field',
            'toss_winner_is_t1': 1, 'toss_decision_field': 1
        },
        {
            'name': 'Scenario B: GT wins toss and bowls first (RCB defending)',
            'toss_winner': 'GT', 'toss_decision': 'field',
            'toss_winner_is_t1': 0, 'toss_decision_field': 1
        },
        {
            'name': 'Scenario C: RCB wins toss and bats first (RCB defending)',
            'toss_winner': 'RCB', 'toss_decision': 'bat',
            'toss_winner_is_t1': 1, 'toss_decision_field': 0
        },
        {
            'name': 'Scenario D: GT wins toss and bats first (Chasing)',
            'toss_winner': 'GT', 'toss_decision': 'bat',
            'toss_winner_is_t1': 0, 'toss_decision_field': 0
        }
    ]
    
    print_section("PREDICTING IPL 2026 QUALIFIER 1: RCB vs GT")
    print(f"Venue: {venue}")
    print(f"Current Team Forms (At Playoff Eve):")
    print(f"  - RCB Recent Form (Last 5): {rf_1 * 100:.1f}% Win Rate | Momentum Score: {mom_1:.2f}")
    print(f"  - GT Recent Form (Last 5):  {rf_2 * 100:.1f}% Win Rate | Momentum Score: {mom_2:.2f}")
    print(f"  - Head-to-Head history:     RCB {t1_h2h_rate*100:.1f}% - GT {t2_h2h_rate*100:.1f}%")
    print(f"  - Venue Experience here:    RCB wins {ven_1*100:.1f}% | GT wins {ven_2*100:.1f}%")
    print(f"  - Strategy Success Rate:")
    print(f"    - RCB: Chasing Success = {ch_1*100:.1f}% | Defending Success = {def_1*100:.1f}%")
    print(f"    - GT:  Chasing Success = {ch_2*100:.1f}% | Defending Success = {def_2*100:.1f}%")
    
    print("\n[THEORY COMMENT] HOW THE MODEL OUTPUTS PROBABILITIES")
    print("---------------------------------------------------------")
    print("In a RandomForestClassifier, predictions aren't just a simple yes/no. Each of our 300")
    print("decision trees traverses down its paths to a leaf node. At that leaf node, there is a class")
    print("distribution (e.g., 60% class 1, 40% class 0). The Random Forest averages these class")
    print("distributions across all 300 trees to yield the final ensemble win probability!")
    
    results = []
    
    for sc in scenarios:
        feat_dict = {
            'venue_encoded': [venue_encoded],
            'recent_form_team1': [rf_1],
            'recent_form_team2': [rf_2],
            'momentum_team1': [mom_1],
            'momentum_team2': [mom_2],
            'h2h_team1': [t1_h2h_rate],
            'h2h_team2': [t2_h2h_rate],
            'venue_advantage_team1': [ven_1],
            'venue_advantage_team2': [ven_2],
            'chasing_success_team1': [ch_1],
            'chasing_success_team2': [ch_2],
            'defending_success_team1': [def_1],
            'defending_success_team2': [def_2],
            'toss_winner_is_team1': [sc['toss_winner_is_t1']],
            'toss_decision_field': [sc['toss_decision_field']]
        }
        
        X_pred = pd.DataFrame(feat_dict)
        
        proba = rf_model.predict_proba(X_pred)
        rcb_prob = proba[0][1]
        gt_prob = proba[0][0]
        
        if rcb_prob > gt_prob:
            predicted_winner = 'RCB'
            winner_prob = rcb_prob
            loser_prob = gt_prob
        else:
            predicted_winner = 'GT'
            winner_prob = gt_prob
            loser_prob = rcb_prob
            
        if winner_prob >= 0.60:
            confidence = 'High'
        elif winner_prob >= 0.53:
            confidence = 'Medium'
        else:
            confidence = 'Low'
            
        factors = []
        
        if rf_1 > rf_2:
            factors.append(f"- RCB stronger overall recent form ({rf_1*100:.0f}% vs {rf_2*100:.0f}%)")
        else:
            factors.append(f"- GT stronger overall recent form ({rf_2*100:.0f}% vs {rf_1*100:.0f}%)")
            
        if mom_1 > mom_2:
            factors.append(f"- RCB exhibits better recent momentum score ({mom_1:.2f} vs {mom_2:.2f})")
        else:
            factors.append(f"- GT exhibits better recent momentum score ({mom_2:.2f} vs {mom_1:.2f})")
            
        if t1_h2h_rate > t2_h2h_rate:
            factors.append(f"- RCB leads head-to-head records ({t1_h2h_rate*100:.0f}% vs {t2_h2h_rate*100:.0f}%)")
        else:
            factors.append(f"- GT leads head-to-head records ({t2_h2h_rate*100:.0f}% vs {t1_h2h_rate*100:.0f}%)")
            
        is_chasing_rcb = (sc['toss_winner'] == 'RCB' and sc['toss_decision'] == 'field') or (sc['toss_winner'] == 'GT' and sc['toss_decision'] == 'bat')
        
        if is_chasing_rcb:
            factors.append(f"- Chasing Scenario: Venue advantage & chasing records favor RCB ({ch_1*100:.0f}% success rate)")
            if ch_1 > def_2:
                factors.append("- RCB chasing success rate is higher than GT defending rate")
        else:
            factors.append(f"- Chasing Scenario: Venue advantage & chasing records favor GT ({ch_2*100:.0f}% success rate)")
            if ch_2 > def_1:
                factors.append("- GT chasing success rate is higher than RCB defending rate")
                
        results.append({
            'scenario': sc['name'],
            'winner': predicted_winner,
            'rcb_prob': rcb_prob,
            'gt_prob': gt_prob,
            'confidence': confidence,
            'factors': factors
        })
        
    for idx, r in enumerate(results):
        print(f"\nSCENARIO {idx + 1}: {r['scenario']}")
        print("=" * 60)
        print(f"Predicted Winner: {r['winner']}")
        print(f"RCB Win Probability: {r['rcb_prob'] * 100:.1f}%")
        print(f"GT Win Probability:  {r['gt_prob'] * 100:.1f}%")
        print(f"Confidence Level:    {r['confidence']}")
        print("\nKey Factors Driving Prediction:")
        for factor in r['factors'][:4]:
            print(f"  {factor}")
        print("=" * 60)

    return results
