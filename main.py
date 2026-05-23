import os
import pandas as pd
from utils.helpers import generate_dataset, print_section, explain_theory
from feature_engineering.features import calculate_rolling_features
from preprocessing.preprocess import preprocess_data
from training.train_model import train_and_evaluate
from prediction.predict_match import predict_qualifier_1

def main():
    print_section("STARTING END-TO-END MACHINE LEARNING PIPELINE")
    
    dataset_path = 'dataset/rcb_gt_matches.csv'
    model_dir = 'model/'
    
    print_section("Step 1: Historical Dataset Generation")
    print("Generating comprehensive, realistic matches for RCB and GT from 2022 to 2026...")
    df = generate_dataset(dataset_path)
    
    print_section("Step 2: Dynamic Rolling Feature Engineering")
    print("Calculating chronological rolling features (Recent Form, H2H, Venue Advantage, Chasing Success)...")
    explain_theory('data_leakage')
    df_feat = calculate_rolling_features(df)
    print(f"Feature engineering complete! Total features engineered: {len(df_feat.columns) - len(df.columns)}")
    
    print_section("Steps 3 & 4: Data Preprocessing & Categorical Encoding")
    print("Imputing missing values, encoding venues, and setting up features (X) and target (y)...")
    X, y = preprocess_data(df_feat, fit_encoders=True, encoders_dir=model_dir)
    
    train_and_evaluate(X, y, model_dir=model_dir)
    
    predict_qualifier_1(df, model_dir=model_dir, venue='HPCA Stadium, Dharamshala')
    
    print_section("PIPELINE COMPLETED SUCCESSFULLY!")
    print("The model has learned all patterns, evaluated its accuracy, and generated predictions.")
    print("You can view all the code inside the rcb_gt_predictor/ folder in your IDE!")
    print("To run this pipeline manually, execute: python main.py")

if __name__ == "__main__":
    main()
