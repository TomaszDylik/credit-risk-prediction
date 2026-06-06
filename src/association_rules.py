"""
association_rules.py
--------------------
Analiza reguł asocjacyjnych na danych kredytowych.

Używa algorytmu Apriori (biblioteka mlxtend) do wyciągnięcia wzorców
wskazujących na brak spłaty kredytu (loan_DEFAULT).

Eksportuje funkcję:
  - run_association_rules(df)  ->  pd.DataFrame z regułami
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# ------------------------------------------------------------------
# STAŁE
# ------------------------------------------------------------------

# Minimalna częstość danego wzorca w danych (5% = co najmniej 5% transakcji)
MIN_SUPPORT = 0.05

# Minimalna pewność reguły – "jeśli X, to DEFAULT z co najmniej 50% pewnością"
MIN_CONFIDENCE = 0.50

# Pokazujemy tylko reguły, gdzie LIFT > 1 (czyli X naprawdę zwiększa ryzyko)
MIN_LIFT = 1.0

# Ile topowych reguł wydrukować
TOP_N = 5


# ------------------------------------------------------------------
# FUNKCJA GŁÓWNA
# ------------------------------------------------------------------

def run_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przyjmuje DataFrame z load_for_rules() (kolumny z etykietami tekstowymi).
    Zwraca posortowaną tabelę reguł asocjacyjnych prowadzących do loan_DEFAULT.

    Kroki:
      1. One-Hot Encoding całego DataFrame (zamieniamy etykiety na kolumny 0/1)
      2. Algorytm Apriori -> zbiory częste (frequent itemsets)
      3. Generowanie reguł asocjacyjnych
      4. Filtrowanie: tylko reguły z consequent = loan_DEFAULT
      5. Sortowanie po lift i confidence
    """

    print("[association_rules] Budowanie macierzy transakcyjnej (OHE)...")

    # Zamieniamy każdą wartość tekstową w osobną kolumnę binarną
    basket = pd.get_dummies(df.astype(str), dtype=bool)

    print(f"[association_rules] Macierz: {basket.shape[0]} wierszy x {basket.shape[1]} kolumn")

    # --- Krok 1: Apriori – szukamy zbiorów, które pojawiają się często ---
    print(f"[association_rules] Uruchamiam Apriori (min_support={MIN_SUPPORT})...")

    frequent_itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        verbose=0,
    )

    print(f"[association_rules] Znaleziono {len(frequent_itemsets)} zbiorów częstych")

    if frequent_itemsets.empty:
        print("[association_rules] UWAGA: brak zbiorów częstych. Obniż MIN_SUPPORT.")
        return pd.DataFrame()

    # --- Krok 2: Generowanie reguł asocjacyjnych ---
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
        num_itemsets=len(frequent_itemsets),
    )

    print(f"[association_rules] Wygenerowano {len(rules)} reguł (min_confidence={MIN_CONFIDENCE})")

    if rules.empty:
        print("[association_rules] UWAGA: brak reguł. Obniż MIN_CONFIDENCE.")
        return pd.DataFrame()

    # --- Krok 3: Filtrujemy tylko reguły prowadzące do DEFAULT ---
    default_label = "loan_status_label_loan_DEFAULT"

    default_rules = rules[
        rules["consequents"].apply(lambda x: default_label in x)
    ].copy()

    print(f"[association_rules] Reguły wskazujące na DEFAULT: {len(default_rules)}")

    # Usuwamy reguły, gdzie LIFT <= 1 (brak faktycznej zależności)
    default_rules = default_rules[default_rules["lift"] > MIN_LIFT]

    if default_rules.empty:
        print("[association_rules] Brak reguł z lift > 1. Sprawdź dane lub obniż progi.")
        return pd.DataFrame()

    # --- Krok 4: Sortowanie – najsilniejsze reguły na górze ---
    default_rules = default_rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False,
    ).reset_index(drop=True)

    return default_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(TOP_N)


# ------------------------------------------------------------------
# WYDRUK WYNIKÓW
# ------------------------------------------------------------------

def _format_frozenset(fs) -> str:
    """Zamienia frozenset na czytelny string."""
    return " + ".join(sorted(str(x).replace("_", " ") for x in fs))


def print_rules(rules_df: pd.DataFrame) -> None:
    """Drukuje reguły w czytelnym formacie do prezentacji."""

    if rules_df.empty:
        print("\n[!] Brak reguł do wydruku.")
        return

    separator = "=" * 72

    print(f"\n{separator}")
    print(" REGUŁY ASOCJACYJNE – wzorce wskazujące na brak spłaty kredytu")
    print(separator)

    for rank, row in rules_df.iterrows():
        antecedent = _format_frozenset(row["antecedents"])
        consequent = _format_frozenset(row["consequents"])

        print(f"\n#{rank + 1}")
        print(f"  JEŚLI:  {antecedent}")
        print(f"  WTEDY:  {consequent}")
        print(f"  Support:    {row['support']:.3f}  "
              f"({row['support'] * 100:.1f}% klientów ma ten wzorzec)")
        print(f"  Confidence: {row['confidence']:.3f}  "
              f"(w {row['confidence'] * 100:.1f}% przypadków prowadzi do DEFAULT)")
        print(f"  Lift:       {row['lift']:.3f}  "
              f"(ryzyko jest {row['lift']:.1f}x wyższe niż losowo)")
 
    print(f"\n{separator}")
    print(" Jak czytać metryki:")
    print("  Support     = jak często dany wzorzec pojawia się w danych")
    print("  Confidence  = jeśli warunek zachodzi, z jakim prawdopodobieństwem")
    print("                nastąpi DEFAULT")
    print("  Lift > 1    = reguła jest nieprzypadkowa, X naprawdę zwiększa ryzyko")
    print(separator)
