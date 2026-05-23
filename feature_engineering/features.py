import pandas as pd
import numpy as np

def calculate_rolling_features(df):
    recent_form_t1, recent_form_t2 = [], []
    momentum_t1, momentum_t2 = [], []
    h2h_t1, h2h_t2 = [], []
    venue_adv_t1, venue_adv_t2 = [], []
    chase_success_t1, chase_success_t2 = [], []
    defend_success_t1, defend_success_t2 = [], []
    
    toss_winner_is_t1 = []
    toss_decision_field = []
    
    weights_5 = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
    
    for idx in range(len(df)):
        current_match = df.iloc[idx]
        current_date = current_match['date']
        t1 = current_match['team1']
        t2 = current_match['team2']
        venue = current_match['venue']
        toss_winner = current_match['toss_winner']
        toss_decision = current_match['toss_decision']
        
        historical_df = df[df['date'] < current_date].copy()
        
        def get_team_metrics(team):
            team_matches = historical_df[(historical_df['team1'] == team) | (historical_df['team2'] == team)]
            team_matches = team_matches.sort_values(by='date', ascending=False)
            
            if len(team_matches) == 0:
                return 0.5, 0.5, 0.5, 0.5, 0.5
            
            last_5 = team_matches.head(5)
            wins_in_last_5 = sum(last_5['winner'] == team)
            recent_form_score = wins_in_last_5 / len(last_5)
            
            momentum_components = []
            for m_idx, (_, match) in enumerate(last_5.iterrows()):
                is_win = match['winner'] == team
                match_score = 0.0
                
                if is_win:
                    if match['win_by_runs'] > 25 or match['win_by_wickets'] > 5:
                        match_score = 1.2
                    else:
                        match_score = 1.0
                else:
                    is_close_loss = False
                    team_batted_first = (
                        (match['toss_winner'] == team and match['toss_decision'] == 'bat') or
                        (match['toss_winner'] != team and match['toss_decision'] == 'field')
                    )
                    
                    if team_batted_first and match['win_by_wickets'] > 0 and match['win_by_wickets'] <= 3:
                        is_close_loss = True
                    elif not team_batted_first and match['win_by_runs'] > 0 and match['win_by_runs'] <= 10:
                        is_close_loss = True
                        
                    match_score = 0.2 if is_close_loss else 0.0
                
                momentum_components.append(match_score)
            
            n_avail = len(momentum_components)
            sub_weights = weights_5[:n_avail]
            sub_weights = sub_weights / np.sum(sub_weights)
            weighted_momentum = float(np.sum(np.array(momentum_components) * sub_weights))
            
            venue_matches = team_matches[team_matches['venue'] == venue]
            if len(venue_matches) == 0:
                venue_advantage = 0.5
            else:
                venue_advantage = sum(venue_matches['winner'] == team) / len(venue_matches)
                
            chase_matches = []
            defend_matches = []
            
            for _, match in team_matches.iterrows():
                bat_second = (
                    (match['toss_winner'] == team and match['toss_decision'] == 'field') or
                    (match['toss_winner'] != team and match['toss_decision'] == 'bat')
                )
                if bat_second:
                    chase_matches.append(match)
                else:
                    defend_matches.append(match)
            
            chase_df = pd.DataFrame(chase_matches)
            defend_df = pd.DataFrame(defend_matches)
            
            chase_rate = sum(chase_df['winner'] == team) / len(chase_df) if len(chase_df) > 0 else 0.5
            defend_rate = sum(defend_df['winner'] == team) / len(defend_df) if len(defend_df) > 0 else 0.5
            
            return recent_form_score, weighted_momentum, venue_advantage, chase_rate, defend_rate

        rf_1, mom_1, ven_1, ch_1, def_1 = get_team_metrics(t1)
        rf_2, mom_2, ven_2, ch_2, def_2 = get_team_metrics(t2)
        
        recent_form_t1.append(rf_1)
        recent_form_t2.append(rf_2)
        momentum_t1.append(mom_1)
        momentum_t2.append(mom_2)
        venue_adv_t1.append(ven_1)
        venue_adv_t2.append(ven_2)
        chase_success_t1.append(ch_1)
        chase_success_t2.append(ch_2)
        defend_success_t1.append(def_1)
        defend_success_t2.append(def_2)
        
        h2h_matches = historical_df[
            ((historical_df['team1'] == t1) & (historical_df['team2'] == t2)) |
            ((historical_df['team1'] == t2) & (historical_df['team2'] == t1))
        ]
        
        if len(h2h_matches) == 0:
            h2h_t1.append(0.5)
            h2h_t2.append(0.5)
        else:
            t1_h2h_wins = sum(h2h_matches['winner'] == t1)
            h2h_t1_rate = t1_h2h_wins / len(h2h_matches)
            h2h_t1.append(h2h_t1_rate)
            h2h_t2.append(1.0 - h2h_t1_rate)
            
        toss_winner_is_t1.append(1 if toss_winner == t1 else 0)
        toss_decision_field.append(1 if toss_decision == 'field' else 0)

    df_feat = df.copy()
    
    df_feat['recent_form_team1'] = recent_form_t1
    df_feat['recent_form_team2'] = recent_form_t2
    df_feat['momentum_team1'] = momentum_t1
    df_feat['momentum_team2'] = momentum_t2
    df_feat['h2h_team1'] = h2h_t1
    df_feat['h2h_team2'] = h2h_t2
    df_feat['venue_advantage_team1'] = venue_adv_t1
    df_feat['venue_advantage_team2'] = venue_adv_t2
    df_feat['chasing_success_team1'] = chase_success_t1
    df_feat['chasing_success_team2'] = chase_success_t2
    df_feat['defending_success_team1'] = defend_success_t1
    df_feat['defending_success_team2'] = defend_success_t2
    
    df_feat['toss_winner_is_team1'] = toss_winner_is_t1
    df_feat['toss_decision_field'] = toss_decision_field
    
    return df_feat
