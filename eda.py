"""
eda.py
Generates 10 EDA charts and saves them to reports/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("/home/claude/Retail_Churn_Project/reports", exist_ok=True)

CLEAN    = "/home/claude/Retail_Churn_Project/data/cleaned_transactions.csv"
FEATURES = "/home/claude/Retail_Churn_Project/data/customer_features.csv"
OUT      = "/home/claude/Retail_Churn_Project/reports"

df   = pd.read_csv(CLEAN,    parse_dates=["Invoice_Date"])
feat = pd.read_csv(FEATURES, parse_dates=["Last_Purchase_Date"])

PALETTE = {"Active":"#2E75B6","Churned":"#E74C3C"}
sns.set_style("whitegrid")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})

# ── 1. Churn Distribution ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2, figsize=(12,5))
fig.suptitle("1. Churn Distribution", fontsize=14, fontweight="bold")
labels = ["Active (0)","Churned (1)"]
sizes  = feat["Churn"].value_counts().sort_index()
axes[0].pie(sizes, labels=labels, autopct="%1.1f%%",
            colors=["#2E75B6","#E74C3C"], startangle=140, wedgeprops={"edgecolor":"white"})
axes[0].set_title("Pie Chart")
sns.countplot(x="Churn", data=feat, palette=["#2E75B6","#E74C3C"], ax=axes[1])
axes[1].set_xticklabels(["Active","Churned"])
axes[1].set_title("Count Plot")
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()+10),
                     ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 1 saved")

# ── 2. Revenue Distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1,2, figsize=(13,5))
fig.suptitle("2. Revenue Distribution by Churn Status", fontsize=14, fontweight="bold")
for label, group in feat.groupby("Churn"):
    tag = "Churned" if label else "Active"
    axes[0].hist(np.log1p(group["Total_Revenue"]), bins=40, alpha=0.6,
                 label=tag, color=PALETTE[tag])
axes[0].set_xlabel("Log(Total Revenue)")
axes[0].set_title("Revenue Distribution (log scale)")
axes[0].legend()
sns.boxplot(x="Churn", y="Total_Revenue", data=feat,
            palette=["#2E75B6","#E74C3C"], ax=axes[1])
axes[1].set_xticklabels(["Active","Churned"])
axes[1].set_yscale("log")
axes[1].set_title("Revenue Box Plot (log scale)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_revenue_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 2 saved")

# ── 3. RFM Segmentation ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1,3, figsize=(15,5))
fig.suptitle("3. RFM Analysis", fontsize=14, fontweight="bold")
for ax, col, color, label in zip(axes,
        ["Recency","Frequency","Monetary"],
        ["#E74C3C","#2E75B6","#27AE60"],
        ["Days Since Last Purchase","Number of Orders","Total Spend (₹)"]):
    axes[list(axes).index(ax)].hist(feat[col], bins=40, color=color, edgecolor="white", alpha=0.85)
    axes[list(axes).index(ax)].set_xlabel(label)
    axes[list(axes).index(ax)].set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT}/03_rfm_segmentation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 3 saved")

# ── 4. Monthly Sales Trend ────────────────────────────────────────────────────
monthly = df.set_index("Invoice_Date").resample("ME")["Total_Amount"].sum().reset_index()
monthly.columns = ["Month","Revenue"]
fig, ax = plt.subplots(figsize=(13,5))
ax.plot(monthly["Month"], monthly["Revenue"], marker="o", color="#2E75B6", linewidth=2)
ax.fill_between(monthly["Month"], monthly["Revenue"], alpha=0.15, color="#2E75B6")
ax.set_title("4. Monthly Sales Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Revenue (₹)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/04_monthly_sales_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 4 saved")

# ── 5. CLV Distribution ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(np.log1p(feat["CLV"]), bins=50, kde=True, color="#8E44AD", ax=ax)
ax.set_title("5. Customer Lifetime Value Distribution (log scale)", fontsize=14, fontweight="bold")
ax.set_xlabel("Log(CLV)")
plt.tight_layout()
plt.savefig(f"{OUT}/05_clv_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 5 saved")

# ── 6. Purchase Frequency ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(feat["Frequency"].clip(0,50), bins=40, kde=True, color="#E67E22", ax=ax)
ax.set_title("6. Purchase Frequency Analysis", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Transactions")
plt.tight_layout()
plt.savefig(f"{OUT}/06_purchase_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 6 saved")

# ── 7. Product Category Analysis ─────────────────────────────────────────────
cat_rev = df.groupby("Category")["Total_Amount"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11,5))
cat_rev.plot(kind="bar", color="#2E75B6", edgecolor="white", ax=ax)
ax.set_title("7. Revenue by Product Category", fontsize=14, fontweight="bold")
ax.set_ylabel("Total Revenue (₹)")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/07_category_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 7 saved")

# ── 8. Country-wise Churn Rate ────────────────────────────────────────────────
country_cols = [c for c in feat.columns if c.startswith("Country_")]
if country_cols:
    churn_by_country = {}
    for col in country_cols:
        name = col.replace("Country_","")
        sub = feat[feat[col]==1]
        if len(sub) >= 20:
            churn_by_country[name] = sub["Churn"].mean()*100
    c_df = pd.Series(churn_by_country).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11,5))
    c_df.plot(kind="bar", color="#E74C3C", edgecolor="white", ax=ax)
    ax.set_title("8. Country-wise Churn Rate (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/08_country_churn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart 8 saved")

# ── 9. Correlation Heatmap ────────────────────────────────────────────────────
num_cols = ["Recency","Frequency","Monetary","RFM_Score","Avg_Order_Value",
            "CLV","Days_Since_Last_Purchase","Loyalty_Score",
            "Weekend_Ratio","Quarterly_Spend","Churn"]
num_cols = [c for c in num_cols if c in feat.columns]
corr = feat[num_cols].corr()
fig, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size":9})
ax.set_title("9. Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/09_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 9 saved")

# ── 10. Churn vs Key Features ─────────────────────────────────────────────────
fig, axes = plt.subplots(2,2, figsize=(13,10))
fig.suptitle("10. Churn vs Key Features", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flat, ["Recency","Monetary","Loyalty_Score","Avg_Order_Value"]):
    sns.boxplot(x="Churn", y=col, data=feat, palette=["#2E75B6","#E74C3C"], ax=ax)
    ax.set_xticklabels(["Active","Churned"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT}/10_churn_vs_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart 10 saved")

print("\nAll 10 EDA charts saved to reports/")
