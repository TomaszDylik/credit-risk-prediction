# Credit Risk Scoring & ROI Optimization

This project implements a Machine Learning pipeline to assess credit risk and maximize a bank's operational profit. Instead of standard 0/1 predictions, it builds a continuous scoring engine to find the mathematical peak of profitability.

## Key Highlights
* **Preprocessing:** Cleaned ~32k records, applied One-Hot Encoding, and used **SMOTE** to balance the highly skewed 80/20 target distribution.
* **Business Logic:** Validated dataset patterns using the **Apriori** algorithm (e.g., Renting + Low Income + High Burden -> 63.6% default chance).
* **Modeling:** **XGBoost** outperformed classic models (RF, kNN, NB) due to its sequential learning, effectively minimizing False Positives.
* **Financial Simulation:** Proved that the standard 50% probability threshold is financially suboptimal. The custom ROI script dynamically calculated that a strict **~26% Risk Score cutoff** maximizes the net operational profit by heavily reducing capital loss.

## Tech Stack
Python, pandas, scikit-learn, xgboost, imbalanced-learn, mlxtend, matplotlib

## How to Run
```bash
pip install -r requirements.txt
python main.py