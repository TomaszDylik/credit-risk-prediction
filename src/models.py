"""
models.py
---------
Trenowanie i ewaluacja modeli klasyfikacyjnych dla credit risk prediction.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from src.visualizations import (
    plot_results,
    plot_risk_score_distribution,
    plot_business_profit_curve,
    plot_business_confusion_matrix,
    plot_models_comparison,
)

RANDOM_STATE = 42
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # ------------------------------------------------------------------
    # SMOTE – balansowanie zbioru treningowego
    # ------------------------------------------------------------------
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n[models] Rozkład klas po SMOTE (zbiór treningowy):")
    counts = pd.Series(y_train_res).value_counts().sort_index()
    for cls, cnt in counts.items():
        print(f"  Klasa {cls}: {cnt} próbek")

    # ------------------------------------------------------------------
    # Definicja modeli – XGBoost dodany jako piąty klasyfikator
    # ------------------------------------------------------------------
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN":          KNeighborsClassifier(),
        "GaussianNB":   GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        # use_label_encoder=False i eval_metric='logloss' wyciszają
        # ostrzeżenia deprecacji w nowszych wersjach XGBoost
        "XGBoost":      XGBClassifier(
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                        ),
    }

    rf_model  = None   # bazowy RandomForest – potrzebny do Feature Importance
    xgb_model = None   # XGBoost – potrzebny do Feature Importance
    model_recalls = {}

    # ------------------------------------------------------------------
    # Trening, predykcja i ewaluacja – pętla wspólna dla wszystkich modeli
    # ------------------------------------------------------------------
    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"  Model: {name}")
        print("=" * 60)

        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        # W bankowości kluczowa jest miara Recall dla klasy 1 (default),
        # ponieważ koszt przeoczenia prawdziwego defaultu (FN) jest znacznie
        # wyższy niż koszt fałszywego alarmu (FP). Chcemy minimalizować FN.
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Brak defaultu (0)", "Default (1)"]))

        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

        current_recall = f1_score(y_test, y_pred, pos_label=1)
        model_recalls[name] = current_recall

        if name == "RandomForest":
            rf_model = model

        if name == "XGBoost":
            xgb_model = model

    # wykres porównania modeli
    plot_models_comparison(model_recalls)

    # ------------------------------------------------------------------
    # Feature Importance z bazowego RandomForest (Top 5)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Feature Importance – RandomForest (Top 5)")
    print("=" * 60)

    rf_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    rf_top5 = rf_importances.sort_values(ascending=False).head(5)

    for feature, score in rf_top5.items():
        print(f"  {feature:<35} {score:.4f}")

    # ------------------------------------------------------------------
    # Feature Importance z XGBoost (Top 5)
    # Porównanie pozwala ocenić, na które cechy zwracają uwagę oba modele
    # i czy wybór cech jest spójny między algorytmami.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Feature Importance – XGBoost (Top 5)")
    print("=" * 60)

    xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    xgb_top5 = xgb_importances.sort_values(ascending=False).head(5)

    for feature, score in xgb_top5.items():
        print(f"  {feature:<35} {score:.4f}")

    # ------------------------------------------------------------------
    # System Scoringowy (Risk Score)
    # Prawdopodobieństwo defaultu z XGBoost skalowane do zakresu 0–100.
    # Trzy koszyki ryzyka odzwierciedlają praktykę bankowego scoringu
    # kredytowego: Niskie (0–20%), Średnie (20–60%), Wysokie (>60%).
    # ------------------------------------------------------------------
    THRESHOLD = 0.5

    risk_scores = xgb_model.predict_proba(X_test)[:, 1] * 100

    low    = (risk_scores <= 20).sum()
    medium = ((risk_scores > 20) & (risk_scores <= 60)).sum()
    high   = (risk_scores > 60).sum()
    total  = len(risk_scores)

    print("\n" + "=" * 60)
    print("  XGBoost – System Scoringowy (Risk Score 0–100%)")
    print("=" * 60)
    print(f"\n  Niskie ryzyko   ( 0–20%): {low:>6}  ({low / total * 100:.1f}%)")
    print(f"  Średnie ryzyko  (20–60%): {medium:>6}  ({medium / total * 100:.1f}%)")
    print(f"  Wysokie ryzyko  (>60%):   {high:>6}  ({high / total * 100:.1f}%)")
    print(f"  {'─' * 38}")
    print(f"  Łącznie:                  {total:>6}")

    y_pred_xgb = (risk_scores / 100 >= THRESHOLD).astype(int)

    print(f"\nClassification Report (próg {THRESHOLD}):")
    print(
        classification_report(
            y_test,
            y_pred_xgb,
            target_names=["Brak defaultu (0)", "Default (1)"],
        )
    )

    print("Confusion Matrix:")
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    print(f"  TN={cm_xgb[0,0]}  FP={cm_xgb[0,1]}")
    print(f"  FN={cm_xgb[1,0]}  TP={cm_xgb[1,1]}")

    # ------------------------------------------------------------------
    # Generowanie wykresów
    # ------------------------------------------------------------------
    plot_risk_score_distribution(y_test, risk_scores)

    plot_results(
        y_true=y_test,
        y_pred_xgb=y_pred_xgb,
        importances_xgb=xgb_model.feature_importances_,
        feature_names=X_train.columns,
        threshold_label=str(THRESHOLD),
    )

    # ------------------------------------------------------------------
    # FINAŁ BIZNESOWY: Symulacja Zysków Banku na XGBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Wdrożenie Biznesowe: Symulacja Optymalnego Portfela")
    print("=" * 60)
    
    # 1. Dynamiczne wyliczenie średniej kwoty kredytu z surowych danych
    # (Omijamy X_test, ponieważ dane tam są przeskalowane przez StandardScaler)
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv_path = os.path.join(base_dir, "data", "raw", "credit_risk_dataset.csv")

    bank_commission = 0.30
    
    try:
        df_raw = pd.read_csv(raw_csv_path)
        dynamic_avg_loan = df_raw['loan_amnt'].mean()
    except Exception:
        dynamic_avg_loan = 9583.0 # Wartość awaryjna, gdyby nie znalazł pliku

    print(f"Wyliczona średnia kwota kredytu w portfelu: {dynamic_avg_loan:,.0f} USD")
    
    # 2. Pobieramy dokładne % ryzyka dla każdego klienta z XGBoosta
    y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # 3. Odpalamy symulację z DYNAMICZNĄ kwotą kredytu
    best_thresh, max_profit = plot_business_profit_curve(
        y_test, 
        y_probs_xgb, 
        avg_loan_amount=dynamic_avg_loan, 
        margin=bank_commission
    )
    
    # 4. Generowanie drugiej macierzy pomyłek dla progu biznesowego
    plot_business_confusion_matrix(y_test, y_probs_xgb, best_thresh)
    
    print(f"Najbardziej zyskowny próg odcięcia: {best_thresh*100:.0f}%")
    print(f"Maksymalny wyliczony zysk z portfela testowego: {max_profit:,.0f} USD")
    print("-" * 60)
    print(f"Wniosek Biznesowy dla firmy:")
    print(f"Odrzucaj bezwzględnie każdego wnioskującego,")
    print(f"którego 'Risk Score' wyliczony przez XGBoost wynosi {best_thresh*100:.0f}% lub więcej.")
    print("=" * 60 + "\n")