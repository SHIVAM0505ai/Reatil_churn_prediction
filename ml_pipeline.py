"""
ml_pipeline.py
Trains and evaluates 4 churn prediction models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting (sklearn)

Outputs:
  - Model comparison table
  - ROC curve chart
  - Confusion matrices
  - Feature importance chart
  - Best model saved to models/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, joblib

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (accuracy_score, precision_score, recall_score,
                                        f1_score, roc_auc_score, confusion_matrix,
                                        roc_curve, ConfusionMatrixDisplay)
from sklearn.pipeline          import Pipeline

warnings.filterwarnings("ignore")
os.makedirs("/home/claude/Retail_Churn_Project/models",  exist_ok=True)
os.makedirs("/home/claude/Retail_Churn_Project/reports", exist_ok=True)

FEATURES = "/home/claude/Retail_Churn_Project/data/customer_features.csv"
OUT      = "/home/claude/Retail_Churn_Project/reports"
MODEL_DIR= "/home/claude/Retail_Churn_Project/models"

feat = pd.read_csv(FEATURES)

# ── Features / Target ────────────────────────────────────────────────────────
DROP = ["Customer_ID","Last_Purchase_Date","Cluster","PCA1","PCA2","Segment","Churn"]
DROP = [c for c in DROP if c in feat.columns]

X = feat.drop(columns=DROP).select_dtypes(include=[np.number]).fillna(0)
y = feat["Churn"]

print(f"Feature matrix : {X.shape}")
print(f"Churn rate     : {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, max_depth=10,
                                          random_state=42, class_weight="balanced", n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                               max_depth=4, random_state=42))
    ]),
}

# ── Train & Evaluate ──────────────────────────────────────────────────────────
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows  = []
probs = {}

for name, pipeline in MODELS.items():
    print(f"\nTraining {name} ...")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_prob  = pipeline.predict_proba(X_test)[:,1]
    probs[name] = y_prob

    cv_auc = cross_val_score(pipeline, X_train, y_train,
                              scoring="roc_auc", cv=cv).mean()

    rows.append({
        "Model"        : name,
        "Accuracy"     : accuracy_score(y_test, y_pred),
        "Precision"    : precision_score(y_test, y_pred),
        "Recall"       : recall_score(y_test, y_pred),
        "F1 Score"     : f1_score(y_test, y_pred),
        "ROC-AUC"      : roc_auc_score(y_test, y_prob),
        "CV ROC-AUC"   : cv_auc,
    })

results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
print("\n── Model Comparison ──")
print(results.to_string(index=False))
results.to_csv(f"{OUT}/model_comparison.csv", index=False)

# ── ROC Curves ────────────────────────────────────────────────────────────────
COLORS_ROC = ["#2E75B6","#E74C3C","#27AE60","#8E44AD"]
fig, ax = plt.subplots(figsize=(9,7))
for (name, prob), color in zip(probs.items(), COLORS_ROC):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/14_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("ROC chart saved")

# ── Confusion Matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2,2, figsize=(12,10))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, pipeline) in zip(axes.flat, MODELS.items()):
    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["Active","Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig(f"{OUT}/15_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrix chart saved")

# ── Feature Importance (Random Forest) ───────────────────────────────────────
rf_pipeline   = MODELS["Random Forest"]
rf_clf        = rf_pipeline.named_steps["clf"]
importances   = pd.Series(rf_clf.feature_importances_, index=X.columns)
top20         = importances.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10,8))
top20.plot(kind="barh", color="#2E75B6", edgecolor="white", ax=ax)
ax.set_title("Top 20 Feature Importances — Random Forest", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/16_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance chart saved")

# ── Save best model ───────────────────────────────────────────────────────────
best_name     = results.iloc[0]["Model"]
best_pipeline = MODELS[best_name]
joblib.dump(best_pipeline, f"{MODEL_DIR}/best_churn_model.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_columns.pkl")
print(f"\nBest model : {best_name}")
print(f"Saved      → {MODEL_DIR}/best_churn_model.pkl")

# ── Model comparison bar chart ────────────────────────────────────────────────
metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
r = results.set_index("Model")[metrics]
fig, ax = plt.subplots(figsize=(13,6))
r.T.plot(kind="bar", ax=ax, edgecolor="white")
ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
ax.set_ylim(0.4, 1.0)
ax.set_xticklabels(metrics, rotation=0)
ax.legend(title="Model", bbox_to_anchor=(1.01,1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/17_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Model comparison chart saved")
print("\nML Pipeline complete!")
