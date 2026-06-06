"""
main.py
-------
Punkt wejscia projektu. Uruchamia preprocessing i analize regul asocjacyjnych.

Uzycie:
    python main.py
"""

import sys
import os

# Wymuszamy UTF-8 na stdout – bez tego Windows CP1252 blokuje polskie znaki
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Dodajemy katalog główny projektu do ścieżki Pythona,
# żeby import z src/ działał niezależnie od miejsca uruchomienia
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_preprocess, load_for_rules
from src.association_rules import run_association_rules, print_rules
from src.models import train_and_evaluate_models


def main():
    print("\n" + "=" * 60)
    print("  CREDIT RISK PREDICTION – pipeline")
    print("=" * 60)

    # ----------------------------------------------------------
    # KROK 1: Preprocessing i przygotowanie danych dla modeli ML
    # ----------------------------------------------------------
    print("\n>>> KROK 1: Preprocessing (OHE + skalowanie)")
    print("-" * 60)

    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("\n[main] Podsumowanie po preprocessingu:")
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"  y_train – klasa 0: {(y_train == 0).sum()},  klasa 1: {(y_train == 1).sum()}")
    print(f"  y_test  – klasa 0: {(y_test == 0).sum()},  klasa 1: {(y_test == 1).sum()}")
    print(f"\n  Pierwsze 5 kolumn po OHE: {list(X_train.columns[:5])}")

    # ----------------------------------------------------------
    # KROK 2: Reguły asocjacyjne
    # ----------------------------------------------------------
    print("\n\n>>> KROK 2: Reguły asocjacyjne (Apriori)")
    print("-" * 60)

    rules_input = load_for_rules()
    rules_df = run_association_rules(rules_input)
    print_rules(rules_df)

    # ----------------------------------------------------------
    # KROK 3: Modelowanie (SMOTE + klasyfikatory)
    # ----------------------------------------------------------
    print("\n\n>>> KROK 3: Modelowanie")
    print("-" * 60)

    train_and_evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
