"""
train.py — Model Training, Evaluation & Persistence
Trains Logistic Regression, Naive Bayes, and Random Forest on TF-IDF features.
Saves the best model + vectorizer to disk via joblib.
"""

import os, sys, time, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "news_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
PALETTE = {"bg": "#0d1117", "card": "#161b22", "accent": "#58a6ff",
           "green": "#3fb950", "red": "#f85149", "muted": "#8b949e",
           "text": "#e6edf3", "border": "#30363d"}


def load_data(path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(" STEP 1 — Loading Dataset")
    print(f"{'='*55}")
    if not os.path.exists(path):
        print("  Dataset not found. Generating synthetic dataset…")
        sys.path.insert(0, os.path.join(BASE_DIR, "dataset"))
        from generate_dataset import generate_dataset
        generate_dataset(500, 500, path)
    df = pd.read_csv(path)
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    vc = df["label"].value_counts()
    for lbl, cnt in vc.items():
        print(f"  {lbl:<6} : {cnt:,} ({cnt/len(df)*100:.1f}%)")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(" STEP 2 — Preprocessing")
    print(f"{'='*55}")
    sys.path.insert(0, BASE_DIR)
    from preprocess import preprocess_dataframe
    df = preprocess_dataframe(df, use_stemming=True)
    print(f"  Sample cleaned text:\n    {df['cleaned'].iloc[0][:120]}…")
    return df


def extract_features(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(" STEP 3 — TF-IDF Feature Extraction")
    print(f"{'='*55}")
    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(df["cleaned"])
    y = (df["label"] == "FAKE").astype(int)   # FAKE=1, REAL=0
    print(f"  Vocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"  Feature matrix  : {X.shape[0]:,} × {X.shape[1]:,}")
    print(f"  Top 10 features : {list(vectorizer.vocabulary_.keys())[:10]}")
    return X, y, vectorizer


def train_models(X_train, X_test, y_train, y_test):
    print(f"\n{'='*55}")
    print(" STEP 4 — Model Training")
    print(f"{'='*55}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   solver="lbfgs", random_state=42),
        "Naive Bayes":         MultinomialNB(alpha=0.1),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=20,
                                                      n_jobs=-1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}…", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        cm   = confusion_matrix(y_test, preds)

        results[name] = dict(model=model, acc=acc, prec=prec, rec=rec,
                              f1=f1, cm=cm, preds=preds, time=elapsed)
        print(f"done ({elapsed:.1f}s)")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1-Score  : {f1:.4f}")
        print(classification_report(y_test, preds,
              target_names=["REAL","FAKE"], digits=4))

    return results


def plot_results(results: dict, y_test):
    print(f"\n{'='*55}")
    print(" STEP 5 — Generating Evaluation Plots")
    print(f"{'='*55}")

    names   = list(results.keys())
    metrics = ["acc", "prec", "rec", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = ["#58a6ff", "#3fb950", "#f0883e", "#d2a8ff"]

    # ── 1. Metric comparison bar chart ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE["bg"])

    ax = axes[0]
    ax.set_facecolor(PALETTE["card"])
    x = np.arange(len(names))
    width = 0.2
    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=lbl, color=col, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                    color=PALETTE["text"])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, color=PALETTE["text"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=PALETTE["text"])
    ax.set_title("Model Comparison — All Metrics", color=PALETTE["text"],
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors=PALETTE["muted"])
    ax.spines[:].set_color(PALETTE["border"])
    ax.legend(fontsize=8, facecolor=PALETTE["card"], labelcolor=PALETTE["text"],
              framealpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])

    # ── 2. Accuracy heatmap ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["card"])
    metric_matrix = np.array([[results[n][m] for m in metrics] for n in names])
    im = ax2.imshow(metric_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, color=PALETTE["text"], fontsize=9)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, color=PALETTE["text"], fontsize=9)
    ax2.set_title("Performance Heatmap", color=PALETTE["text"],
                  fontsize=12, fontweight="bold", pad=12)
    for i in range(len(names)):
        for j in range(len(labels)):
            ax2.text(j, i, f"{metric_matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=10, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax2)

    plt.tight_layout(pad=2)
    path1 = os.path.join(STATIC_DIR, "comparison_chart.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path1}")

    # ── 3. Confusion matrices ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=PALETTE["bg"])
    for ax, name in zip(axes, names):
        ax.set_facecolor(PALETTE["card"])
        cm = results[name]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    cmap="Blues", cbar=False,
                    xticklabels=["REAL","FAKE"],
                    yticklabels=["REAL","FAKE"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(name, color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", color=PALETTE["muted"])
        ax.set_ylabel("Actual", color=PALETTE["muted"])
        ax.tick_params(colors=PALETTE["text"])
    plt.suptitle("Confusion Matrices", color=PALETTE["text"],
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path2 = os.path.join(STATIC_DIR, "confusion_matrices.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path2}")

    # ── 4. F1 radar / spider chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    model_colors = ["#58a6ff", "#3fb950", "#f0883e"]
    for (name, res), col in zip(results.items(), model_colors):
        vals = [res["acc"], res["prec"], res["rec"], res["f1"]]
        ax.bar(labels, vals, alpha=0.4, color=col, label=name)
        ax.plot(labels, vals, "o-", color=col, linewidth=2, markersize=6)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color=PALETTE["text"])
    ax.set_title("Model Performance Overview", color=PALETTE["text"],
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[:].set_color(PALETTE["border"])
    ax.legend(fontsize=9, facecolor=PALETTE["card"], labelcolor=PALETTE["text"])
    plt.tight_layout()
    path3 = os.path.join(STATIC_DIR, "performance_overview.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path3}")


def save_best_model(results: dict, vectorizer):
    print(f"\n{'='*55}")
    print(" STEP 6 — Saving Best Model")
    print(f"{'='*55}")
    best_name = max(results, key=lambda n: results[n]["f1"])
    best      = results[best_name]
    print(f"  Best model: {best_name} (F1={best['f1']:.4f})")

    joblib.dump(best["model"],  os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(vectorizer,     os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(best_name,      os.path.join(MODEL_DIR, "best_model_name.pkl"))

    # Save all models
    for name, res in results.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(res["model"], os.path.join(MODEL_DIR, f"{safe}.pkl"))

    # Save metrics summary
    summary = {n: {k: v for k, v in r.items() if k not in ("model","preds","cm")}
               for n, r in results.items()}
    joblib.dump(summary, os.path.join(MODEL_DIR, "metrics_summary.pkl"))
    print(f"  Models saved to: {MODEL_DIR}")
    return best_name


def main():
    print("\n" + "#"*55)
    print("  FAKE NEWS DETECTION — TRAINING PIPELINE")
    print("#"*55)

    df          = load_data(DATA_PATH)
    df          = preprocess(df)
    X, y, vec   = extract_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n  Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    results     = train_models(X_train, X_test, y_train, y_test)
    plot_results(results, y_test)
    best        = save_best_model(results, vec)

    print(f"\n{'='*55}")
    print(f" Training complete! Best model: {best}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
