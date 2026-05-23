import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df_feat, fit_encoders=True, encoders_dir='model/'):
    os.makedirs(encoders_dir, exist_ok=True)
    
    ratio_columns = [
        'recent_form_team1', 'recent_form_team2',
        'momentum_team1', 'momentum_team2',
        'h2h_team1', 'h2h_team2',
        'venue_advantage_team1', 'venue_advantage_team2',
        'chasing_success_team1', 'chasing_success_team2',
        'defending_success_team1', 'defending_success_team2'
    ]
    
    for col in ratio_columns:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(0.5)
            
    venue_encoder_path = os.path.join(encoders_dir, 'venue_encoder.pkl')
    
    if fit_encoders:
        venue_encoder = LabelEncoder()
        df_feat['venue_encoded'] = venue_encoder.fit_transform(df_feat['venue'].astype(str))
        joblib.dump(venue_encoder, venue_encoder_path)
    else:
        if os.path.exists(venue_encoder_path):
            venue_encoder = joblib.load(venue_encoder_path)
            df_feat['venue_encoded'] = df_feat['venue'].apply(
                lambda x: venue_encoder.transform([x])[0] if x in venue_encoder.classes_ else len(venue_encoder.classes_)
            )
        else:
            raise FileNotFoundError(f"LabelEncoder not found at {venue_encoder_path}. Run training first.")
            
    y = (df_feat['winner'] == df_feat['team1']).astype(int)
    
    feature_cols = [
        'venue_encoded',
        'recent_form_team1', 'recent_form_team2',
        'momentum_team1', 'momentum_team2',
        'h2h_team1', 'h2h_team2',
        'venue_advantage_team1', 'venue_advantage_team2',
        'chasing_success_team1', 'chasing_success_team2',
        'defending_success_team1', 'defending_success_team2',
        'toss_winner_is_team1',
        'toss_decision_field'
    ]
    
    X = df_feat[feature_cols].copy()
    
    print("[PREPROCESSING] Completed successfully.")
    print(f"[PREPROCESSING] Feature matrix shape: {X.shape}")
    print(f"[PREPROCESSING] Target variable distribution: Win Team1={sum(y==1)}, Win Team2={sum(y==0)}")
    
    return X, y
