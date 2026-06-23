"""
segmentation.py
K-Means clustering on RFM to create 5 customer segments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("/home/claude/Retail_Churn_Project/reports", exist_ok=True)

FEATURES = "/home/claude/Retail_Churn_Project/data/customer_features.csv"
OUT      = "/home/claude/Retail_Churn_Project/reports"

feat = pd.read_csv(FEATURES)

rfm_cols = ["Recency","Frequency","Monetary","Loyalty_Score","CLV"]
X = feat[rfm_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow Method ──────────────────────────────────────────────────────────────
inertias = []
K_range  = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(K_range, inertias, marker="o", color="#2E75B6", linewidth=2)
ax.set_title("Elbow Method — Optimal K", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
plt.tight_layout()
plt.savefig(f"{OUT}/11_elbow_method.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Final KMeans (K=5) ────────────────────────────────────────────────────────
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
feat["Cluster"] = km5.fit_predict(X_scaled)

# ── Label segments by RFM profile ────────────────────────────────────────────
profile = feat.groupby("Cluster")[rfm_cols + ["Churn"]].mean()
profile["Size"] = feat["Cluster"].value_counts()

# Sort by Recency asc, Frequency desc to assign meaningful labels
profile_sorted = profile.sort_values(["Recency","Frequency"], ascending=[True,False])

SEGMENT_LABELS = {
    profile_sorted.index[0]: "VIP Customers",
    profile_sorted.index[1]: "Loyal Customers",
    profile_sorted.index[2]: "At-Risk Customers",
    profile_sorted.index[3]: "New Customers",
    profile_sorted.index[4]: "Lost Customers",
}
feat["Segment"] = feat["Cluster"].map(SEGMENT_LABELS)

# ── Cluster Visualization (PCA 2D) ────────────────────────────────────────────
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
feat["PCA1"] = X_pca[:,0]
feat["PCA2"] = X_pca[:,1]

COLORS = {
    "VIP Customers"    : "#F1C40F",
    "Loyal Customers"  : "#2E75B6",
    "At-Risk Customers": "#E67E22",
    "New Customers"    : "#27AE60",
    "Lost Customers"   : "#E74C3C",
}

fig, ax = plt.subplots(figsize=(11,7))
for seg, grp in feat.groupby("Segment"):
    ax.scatter(grp["PCA1"], grp["PCA2"], label=seg,
               color=COLORS.get(seg,"gray"), alpha=0.55, s=18)
ax.set_title("Customer Segments (PCA Projection)", fontsize=14, fontweight="bold")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend(title="Segment", markerscale=2)
plt.tight_layout()
plt.savefig(f"{OUT}/12_customer_segments.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Segment Profile Bar ───────────────────────────────────────────────────────
seg_counts = feat["Segment"].value_counts()
fig, ax = plt.subplots(figsize=(10,5))
seg_counts.plot(kind="bar", color=[COLORS[s] for s in seg_counts.index],
                edgecolor="white", ax=ax)
ax.set_title("Customer Count per Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Customers")
plt.xticks(rotation=20, ha="right")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x()+p.get_width()/2, p.get_height()+5), ha='center')
plt.tight_layout()
plt.savefig(f"{OUT}/13_segment_counts.png", dpi=150, bbox_inches="tight")
plt.close()

# Save with segments
feat.to_csv(FEATURES, index=False)

print("Segment Profile:\n")
print(feat.groupby("Segment")[rfm_cols + ["Churn"]].mean().round(2))
print(f"\nSegment sizes:\n{feat['Segment'].value_counts()}")
print("\nSegmentation complete → reports saved")
