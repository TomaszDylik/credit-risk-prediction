"""
visualizations.py
-----------------
Generowanie i zapis wykresów do prezentacji wyników modelu XGBoost.
Wykresy zapisywane są do folderu plots/ w katalogu głównym projektu.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ścieżka do folderu plots/ wyznaczana relatywnie do tego pliku
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

_PALETTE = "#2C7BB6"          # główny kolor wykresu słupkowego
_CMAP    = "Blues"            # mapa kolorów macierzy pomyłek


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[visualizations] Zapisano: {path}")
    plt.close(fig)


def _plot_confusion_matrix(y_true, y_pred, threshold_label: str) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Brak defaultu (0)", "Default (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=_CMAP,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_title(
        f"Macierz pomyłek – XGBoost (próg {threshold_label})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Wartość przewidziana", fontsize=11)
    ax.set_ylabel("Wartość rzeczywista", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    tn, fp, fn, tp = cm.ravel()
    fig.text(
        0.5, -0.03,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        ha="center", fontsize=10, color="gray",
    )
    return fig


def _plot_feature_importance(importances: np.ndarray, feature_names, top_n: int = 10) -> plt.Figure:
    import pandas as pd

    series = pd.Series(importances, index=feature_names)
    top = series.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        top.index[::-1],
        top.values[::-1],
        color=_PALETTE,
        edgecolor="white",
        height=0.65,
    )

    # Wartości na końcach słupków
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9, color="#333333",
        )

    ax.set_title(
        f"Ważność cech – XGBoost (Top {top_n})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Ważność (gain)", fontsize=11)
    ax.set_ylabel("Cecha", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    return fig


def plot_risk_score_distribution(y_true, risk_scores) -> None:
    """
    Rysuje histogram rozkładu Risk Score (0–100) z podziałem na klasy
    klientów: spłacił kredyt (y=0) vs default (y=1).

    Zapisuje plots/risk_score_dist.png.
    """
    y_true = np.asarray(y_true)
    risk_scores = np.asarray(risk_scores)

    color_good    = "#2C7BB6"   # niebieski – klienci bez defaultu
    color_default = "#D7191C"   # czerwony  – klienci z defaultem

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        x=risk_scores[y_true == 0],
        bins=40, binrange=(0, 100),
        color=color_good, alpha=0.65,
        label="Spłacił kredyt (y=0)",
        ax=ax,
        kde=True,
        line_kws={"linewidth": 2},
    )
    sns.histplot(
        x=risk_scores[y_true == 1],
        bins=40, binrange=(0, 100),
        color=color_default, alpha=0.55,
        label="Default (y=1)",
        ax=ax,
        kde=True,
        line_kws={"linewidth": 2},
    )

    ax.axvline(20, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(60, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.text(10,  ax.get_ylim()[1] * 0.97, "Niskie\nryzyko",  ha="center", va="top",
            fontsize=9, color="gray")
    ax.text(40,  ax.get_ylim()[1] * 0.97, "Średnie\nryzyko", ha="center", va="top",
            fontsize=9, color="gray")
    ax.text(80,  ax.get_ylim()[1] * 0.97, "Wysokie\nryzyko", ha="center", va="top",
            fontsize=9, color="gray")

    ax.set_xlabel("Risk Score (%)", fontsize=12)
    ax.set_ylabel("Liczba klientów", fontsize=12)
    ax.set_title(
        "Rozkład Risk Score – XGBoost\n(prawdopodobieństwo defaultu × 100)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlim(0, 100)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _save(fig, "risk_score_dist.png")


def plot_results(
    y_true,
    y_pred_xgb,
    importances_xgb: np.ndarray,
    feature_names,
    threshold_label: str = "0.3",
) -> None:
    """
    Generuje i zapisuje dwa wykresy PNG:
      1. plots/confusion_matrix_xgb.png  – macierz pomyłek XGBoost
      2. plots/feature_importance_xgb.png – Top 10 ważności cech XGBoost

    Parametry
    ----------
    y_true          : rzeczywiste etykiety (array-like)
    y_pred_xgb      : predykcje binarne XGBoost (array-like)
    importances_xgb : tablica feature_importances_ z modelu XGBoost
    feature_names   : nazwy cech (np. X_train.columns)
    threshold_label : etykieta progu wyświetlana w tytule wykresu
    """
    print("\n" + "=" * 60)
    print("  Generowanie wykresów...")
    print("=" * 60)

    fig_cm = _plot_confusion_matrix(y_true, y_pred_xgb, threshold_label)
    _save(fig_cm, "confusion_matrix_xgb.png")

    fig_fi = _plot_feature_importance(importances_xgb, feature_names, top_n=10)
    _save(fig_fi, "feature_importance_xgb.png")

    print("[visualizations] Gotowe. Wykresy w folderze plots/\n")


def plot_business_profit_curve(y_true, y_probs, avg_loan_amount=10000, margin=0.10):
    """
    Wylicza i rysuje krzywą zysku dla banku w zależności od progu odcięcia ryzyka.
    Zakłada:
    - Zysk = 10% ze spłaconego kredytu
    - Strata = 100% kwoty przy braku spłaty
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from sklearn.metrics import confusion_matrix
    import os

    thresholds = np.linspace(0.01, 0.99, 99)
    profits = []
    y_true_arr = np.asarray(y_true)
    
    for t in thresholds:
        # Predykcja dla danego progu: jeśli ryzyko >= t, dajemy 1 (odmowa)
        y_pred = (y_probs >= t).astype(int)
        
        # cm = [[TN, FP], [FN, TP]]
        # Klasa 0 (Dobre), Klasa 1 (Złe)
        # TN (True Negative) - bank dał kredyt i gość spłacił -> ZAROBEK
        # FN (False Negative) - bank dał kredyt a gość uciekł -> STRATA
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Obliczenie zysku całkowitego dla tego progu
        profit = (tn * avg_loan_amount * margin) - (fn * avg_loan_amount)
        profits.append(profit)
        
    profits = np.array(profits)
    
    # Znalezienie najlepszego progu (szczyt góry)
    best_idx = np.argmax(profits)
    best_threshold = thresholds[best_idx]
    max_profit = profits[best_idx]
    
    # --- RYSOWANIE WYKRESU ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, profits, color="#2C7BB6", linewidth=3, label="Całkowity zysk portfela")
    
    # Zaznaczenie szczytu
    ax.axvline(best_threshold, color="#D7191C", linestyle="--", linewidth=2)
    ax.scatter([best_threshold], [max_profit], color="#D7191C", s=120, zorder=5)
    
    # Napisy przy szczycie
    ax.text(best_threshold + 0.03, max_profit * 0.95, 
            f"Optymalne odcięcie: {best_threshold*100:.0f}%\nMaks. Zysk: {max_profit:,.0f} USD", 
            fontsize=12, fontweight='bold', color="#D7191C")
    
    # Upiększanie
    ax.set_title("Symulacja Finansowa: Zysk vs Próg Odcięcia (Scoring)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Próg odcięcia ryzyka (Powyżej tej wartości odrzucamy klienta)", fontsize=12)
    ax.set_ylabel("Całkowity Zysk (USD)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f} USD"))
    ax.axhline(0, color="black", linewidth=1, alpha=0.5) # Linia zera (bankructwo)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Zapis do pliku
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "business_value_curve.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"[visualizations] Zapisano wykres finansowy: {path}")
    
    return best_threshold, max_profit

def plot_business_confusion_matrix(y_true, y_probs, best_threshold):
    """
    Rysuje i zapisuje macierz pomyłek dla optymalnego progu biznesowego.
    """
    import numpy as np
    y_pred_biz = (np.asarray(y_probs) >= best_threshold).astype(int)
    fig = _plot_confusion_matrix(y_true, y_pred_biz, f"{best_threshold*100:.0f}% (Biznesowy)")
    _save(fig, "confusion_matrix_business.png")

def plot_models_comparison(model_metrics: dict):
    """
    Rysuje wykres słupkowy porównujący Recall różnych modeli.
    Wyróżnia na czerwono model XGBoost.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    labels = list(model_metrics.keys())
    recalls = [model_metrics[m] for m in labels]
    
    # Kolorujemy XGBoost na czerwono, a resztę na szaro/niebiesko
    colors = ['#D7191C' if m == 'XGBoost' else '#A9Cce3' for m in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, recalls, color=colors, edgecolor='white', width=0.6)

    ax.set_title("Porównanie Modeli ML: F1-Score (Skuteczność Biznesowa)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("F1-Score (Balans Precyzji i Czułości)", fontsize=11)
    ax.set_ylim([0.0, 1.05]) # Zostawiamy miejsce na etykiety nad słupkami

    # Dodanie wartości procentowych nad słupkami
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 0.02, 
            f"{yval:.2f}", 
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333'
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Zapis do pliku
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "models_comparison.png")
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[visualizations] Zapisano wykres porównania modeli: {path}")