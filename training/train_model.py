import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from utils.helpers import print_section, explain_theory

def train_and_evaluate(X, y, model_dir='model/'):
    os.makedirs(model_dir, exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    explain_theory('features_labels')
    explain_theory('categorical_encoding')
    
    print_section("ML Pipeline Step 5: Train-Test Split")
    print(f"  - Total matches in dataset: {len(X)}")
    print(f"  - Training matches (80%):  {len(X_train)}")
    print(f"  - Testing matches (20%):   {len(X_test)}")
    
    print("\n[THEORY COMMENT] WHY WE SPLIT THE DATA")
    print("-----------------------------------------")
    print("If we test our model on the same data we used to train it, it could simply 'memorize'")
    print("the results (overfitting) and score 100% accuracy. By evaluating on an independent test set,")
    print("we get an honest, unbiased estimation of how our model will perform on the real IPL 2026 Qualifier 1!")
    
    print_section("ML Pipeline Step 6: Training Random Forest Classifier")
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    
    print("Building 300 Decision Trees and training the ensemble...")
    rf_model.fit(X_train, y_train)
    print("Model training complete!")
    
    explain_theory('random_forest')
    
    print_section("ML Pipeline Step 7: Model Evaluation")
    
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Team2 Win', 'Team1 Win'])
    
    print(f"Model Accuracy Score: {acc * 100:.2f}%")
    print("\nCONFUSION MATRIX:")
    print("------------------")
    print(conf_matrix)
    print("\nHow to read the Confusion Matrix:")
    print(f"  - Top-Left ({conf_matrix[0,0]}): True Negatives (Model predicted Team 2 wins, and Team 2 actually won)")
    print(f"  - Bottom-Right ({conf_matrix[1,1]}): True Positives (Model predicted Team 1 wins, and Team 1 actually won)")
    print(f"  - Top-Right ({conf_matrix[0,1]}): False Positives (Model predicted Team 1 wins, but Team 2 actually won)")
    print(f"  - Bottom-Left ({conf_matrix[1,0]}): False Negatives (Model predicted Team 2 wins, but Team 1 actually won)")
    
    print("\nCLASSIFICATION REPORT:")
    print("------------------------")
    print(class_report)
    
    print_section("Feature Importance Analysis")
    
    importances = rf_model.feature_importances_
    features_list = X.columns
    
    feature_mapping = {
        'venue_encoded': 'Venue ID',
        'recent_form_team1': 'Recent Form (Team 1)',
        'recent_form_team2': 'Recent Form (Team 2)',
        'momentum_team1': 'Momentum Score (Team 1)',
        'momentum_team2': 'Momentum Score (Team 2)',
        'h2h_team1': 'Head-to-Head Win Rate (Team 1)',
        'h2h_team2': 'Head-to-Head Win Rate (Team 2)',
        'venue_advantage_team1': 'Venue Advantage (Team 1)',
        'venue_advantage_team2': 'Venue Advantage (Team 2)',
        'chasing_success_team1': 'Chasing Success (Team 1)',
        'chasing_success_team2': 'Chasing Success (Team 2)',
        'defending_success_team1': 'Defending Success (Team 1)',
        'defending_success_team2': 'Defending Success (Team 2)',
        'toss_winner_is_team1': 'Toss Winner Advantage',
        'toss_decision_field': 'Toss Decision (Bowling First)'
    }
    
    feat_imp_df = pd.DataFrame({
        'Raw Feature': features_list,
        'Description': [feature_mapping.get(col, col) for col in features_list],
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print(f"{'Rank':<5} | {'Key Factor Affecting Prediction':<35} | {'Weight (Importance %)':<25}")
    print("-" * 75)
    for index, row in feat_imp_df.iterrows():
        print(f"{index + 1:<5} | {row['Description']:<35} | {row['Importance'] * 100:.2f}%")
        
    print("\nHow Random Forest measures Feature Importance:")
    print("During training, whenever a decision tree splits on a feature, it reduces impurity")
    print("(makes the children nodes more homogeneous). Feature importance is the average decrease")
    print("in impurity caused by a feature across all 300 decision trees in the forest. A higher percentage")
    print("means that factor played a larger role in driving the model's final split decisions.")
    
    print_section("ML Pipeline Step 8: Saving Model")
    model_path = os.path.join(model_dir, 'random_forest.pkl')
    joblib.dump(rf_model, model_path)
    print(f"Saved trained Random Forest model successfully to '{model_path}'.")
    
    return rf_model, feat_imp_df
