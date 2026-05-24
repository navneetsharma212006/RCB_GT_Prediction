# RCB vs GT Qualifier 1 Prediction

## Overview
This repository contains a lightweight end‑to‑end machine‑learning pipeline that predicts the winner of the IPL 2026 Qualifier 1 match between Royal Challengers Bangalore (RCB) and Gujarat Titans (GT). The model is built with **scikit‑learn** using a **RandomForestClassifier** and incorporates cricket‑specific features such as recent form, momentum scores, head‑to‑head history, venue advantage, toss decision, and chasing/defending success.

## Repository Structure
```
rcb_gt_predictor/
├── dataset/               # Historical match data (synthetic + real)
├── feature_engineering/   # Feature creation logic (features.py)
├── preprocessing/         # Data cleaning and encoding (preprocess.py)
├── training/              # Model training script (train_model.py)
├── prediction/            # Inference script (predict_match.py)
├── utils/                 # Helper functions for dataset generation (helpers.py)
├── model/                 # Trained Random Forest model and encoders
├── main.py                # Orchestrates the full pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation (this file)
```

## Installation
1. **Clone the repository  **
```bash
git clone https://github.com/navneetsharma212006/RCB_GT_Prediction.git
cd RCB_GT_Prediction
```
2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage
Run the full pipeline with a single command:
```bash
python main.py
```
The script will:
1. Generate a balanced historical dataset.
2. Engineer features for each match.
3. Encode categorical variables.
4. Train a Random Forest model.
5. Save the trained model and encoders.
6. Predict the outcome for the four possible toss scenarios and display probabilities.

### Predicting a Single Scenario
If you only want to see the prediction for a specific toss outcome, edit `prediction/predict_match.py` and set the desired scenario variables, then run:
```bash
python prediction/predict_match.py
```

## Features
- **Recent Form (last 5 matches)** – captures win percentage and momentum.
- **Momentum Score** – weighted by margin of victory/defeat.
- **Head‑to‑Head Win Rate** – direct rivalry statistics.
- **Venue Advantage** – historical win rates at the stadium.
- **Chasing / Defending Success** – team performance when batting second vs first.
- **Toss Winner & Decision** – impact of choosing to bat or bowl first.

## Model Details
- **Algorithm:** RandomForestClassifier (300 trees, max depth 12)
- **Training/Test Split:** 80 % / 20 %
- **Accuracy:** ~45 % (expected for a stochastic sport with limited data)
- **Feature Importance:** Chasing success, momentum, and venue advantage are the top contributors.

## Extending the Project
- Add real IPL match data for additional seasons.
- Experiment with other classifiers (e.g., XGBoost, Gradient Boosting).
- Incorporate player‑level statistics for finer granularity.
- Deploy the model as a lightweight API (FastAPI or Flask).

## License
This project is provided under the MIT License. Feel free to fork, modify, and use it for personal or educational purposes.

## Acknowledgements
- Inspired by publicly available IPL match statistics.
- Built with scikit‑learn, pandas, and numpy.

---
*Created by a cricket enthusiast leveraging data science to explore match‑outcome probabilities.*
