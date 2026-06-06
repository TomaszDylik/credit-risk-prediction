"""
preprocessing.py
----------------
Wczytanie, czyszczenie i przygotowanie danych credit_risk_dataset.csv.

Eksportuje dwie funkcje:
  - load_and_preprocess()  ->  (X_train, X_test, y_train, y_test)
      wersja z OHE + StandardScaler, gotowa dla modeli ML
  - load_for_rules()  ->  DataFrame
      wersja z binnigiem (dyskretyzacją), gotowa dla algorytmu Apriori
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Ścieżka do pliku CSV względem lokalizacji MAIN.PY (katalog główny)
# ------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_BASE_DIR, "data", "raw", "credit_risk_dataset.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET = "loan_status"


# ------------------------------------------------------------------
# POMOCNICZE
# ------------------------------------------------------------------

def _read_raw() -> pd.DataFrame:
    """Wczytuje surowy CSV i zwraca DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"[preprocessing] Wczytano dane: {df.shape[0]} wierszy, {df.shape[1]} kolumn")
    return df


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa anomalie które są ewidentnie błędami w danych:
      - person_age > 100  (brak ludzi w wieku 150 lat)
      - person_emp_length > 60  (staż pracy dłuższy niż normalne życie zawodowe)
    """
    before = len(df)
    df = df[df["person_age"] <= 100].copy()
    df = df[df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)].copy()
    after = len(df)
    print(f"[preprocessing] Usunieto anomalii: {before - after} wierszy")
    return df


def _impute_medians(df_train: pd.DataFrame,
                    df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputacja medianą dla person_emp_length i loan_int_rate.
    WAŻNE: mediana liczona TYLKO na zbiorze treningowym,
    a potem aplikowana do obu zbiorów – zapobiega wyciekowi danych.
    """
    impute_cols = ["person_emp_length", "loan_int_rate"]
    medians = df_train[impute_cols].median()

    df_train[impute_cols] = df_train[impute_cols].fillna(medians)
    df_test[impute_cols] = df_test[impute_cols].fillna(medians)

    print(f"[preprocessing] Uzupełniono braki w: {impute_cols}")
    print(f"               Mediany (z treningu): {medians.to_dict()}")
    return df_train, df_test


# ------------------------------------------------------------------
# FUNKCJA 1: dane dla modeli ML
# ------------------------------------------------------------------

def load_and_preprocess() -> tuple:
    """
    Zwraca (X_train, X_test, y_train, y_test):
      - X_*  jako pd.DataFrame z OHE + przeskalowanymi kolumnami numerycznymi
      - y_*  jako pd.Series (0/1)

    Kolejność kroków:
      1. Wczytaj CSV
      2. Usuń anomalie
      3. Podziel na train/test (PRZED imputacją – brak data leakage)
      4. Imputacja medianą (mediana z treningu)
      5. One-Hot Encoding kolumn kategorycznych
      6. StandardScaler na kolumnach numerycznych
    """
    df = _read_raw()
    df = _remove_outliers(df)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET].astype(int).copy()

    # --- 3. Podział train/test z zachowaniem proporcji klas ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[preprocessing] Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"[preprocessing] Rozkład klas (train):\n{y_train.value_counts().to_string()}")

    # --- 4. Imputacja ---
    X_train, X_test = _impute_medians(X_train.copy(), X_test.copy())

    # Pozostałe ewentualne braki w kolumnach kategorialnych
    X_train[CATEGORICAL_COLS] = X_train[CATEGORICAL_COLS].fillna("Unknown")
    X_test[CATEGORICAL_COLS] = X_test[CATEGORICAL_COLS].fillna("Unknown")

    # --- 5. One-Hot Encoding ---
    X_train_enc = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, dtype=int, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, dtype=int, drop_first=False)

    # Test musi mieć dokładnie te same kolumny co trening
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    # --- 6. Skalowanie kolumn numerycznych ---
    scaler = StandardScaler()
    X_train_enc[NUMERIC_COLS] = scaler.fit_transform(X_train_enc[NUMERIC_COLS])
    X_test_enc[NUMERIC_COLS] = scaler.transform(X_test_enc[NUMERIC_COLS])

    print(f"[preprocessing] Gotowe. Liczba cech po OHE: {X_train_enc.shape[1]}")
    return X_train_enc, X_test_enc, y_train, y_test


# ------------------------------------------------------------------
# FUNKCJA 2: dane z binnigiem, dla algorytmu Apriori
# ------------------------------------------------------------------

def load_for_rules() -> pd.DataFrame:
    """
    Zwraca DataFrame z:
      - zdyskretyzowanymi zmiennymi numerycznymi (binning)
      - zachowanymi kolumnami kategorycznymi
      - kolumną 'loan_status_label' (paid / default)

    Nie jest tu potrzebny podział train/test – analiza reguł
    jest krokiem eksploracyjnym i wykonujemy ją na całym zbiorze
    (lub opcjonalnie tylko na danych treningowych).
    """
    df = _read_raw()
    df = _remove_outliers(df)

    # Imputacja na całym zbiorze (dla eksploracji to akceptowalne)
    impute_cols = ["person_emp_length", "loan_int_rate"]
    df[impute_cols] = df[impute_cols].fillna(df[impute_cols].median())
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("Unknown")

    rules_df = pd.DataFrame()

    # Dyskretyzacja zmiennych numerycznych do etykiet tekstowych
    rules_df["income_bin"] = pd.qcut(
        df["person_income"], q=3,
        labels=["income_LOW", "income_MED", "income_HIGH"],
        duplicates="drop",
    )
    rules_df["loan_amnt_bin"] = pd.qcut(
        df["loan_amnt"], q=3,
        labels=["loan_LOW", "loan_MED", "loan_HIGH"],
        duplicates="drop",
    )
    rules_df["int_rate_bin"] = pd.qcut(
        df["loan_int_rate"], q=3,
        labels=["rate_LOW", "rate_MED", "rate_HIGH"],
        duplicates="drop",
    )
    rules_df["burden_bin"] = pd.qcut(
        df["loan_percent_income"], q=3,
        labels=["burden_LOW", "burden_MED", "burden_HIGH"],
        duplicates="drop",
    )
    rules_df["emp_bin"] = pd.cut(
        df["person_emp_length"],
        bins=[-0.001, 1, 5, 200],
        labels=["emp_0-1y", "emp_2-5y", "emp_6+y"],
    )

    # Kolumny kategoryczne bez zmian
    for col in CATEGORICAL_COLS:
        rules_df[col] = df[col].astype(str)

    # Etykieta celu
    rules_df["loan_status_label"] = df[TARGET].map({0: "loan_PAID", 1: "loan_DEFAULT"})

    print(f"[preprocessing] Dane do reguł gotowe: {rules_df.shape}")
    return rules_df
