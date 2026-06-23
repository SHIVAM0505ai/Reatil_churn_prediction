"""
feature_engineering.py
Builds a customer-level feature table with:
  - Churn label (target)
  - RFM features
  - Customer metrics (AOV, CLV, retention rate ...)
  - Behavioural features (preferred category, weekend ratio ...)
"""

import pandas as pd
import numpy as np

CLEAN    = "/home/claude/Retail_Churn_Project/data/cleaned_transactions.csv"
FEATURES = "/home/claude/Retail_Churn_Project/data/customer_features.csv"

SNAPSHOT_DATE = pd.Timestamp("2024-12-31")
CHURN_DAYS    = 90   # no purchase in last 90 days → churned

df = pd.read_csv(CLEAN, parse_dates=["Invoice_Date","Last_Purchase_Date"])

# ── Churn label ───────────────────────────────────────────────────────────────
last_purchase = df.groupby("Customer_ID")["Invoice_Date"].max().reset_index()
last_purchase.columns = ["Customer_ID","Last_Purchase_Date"]
last_purchase["Days_Since_Last_Purchase"] = (
    SNAPSHOT_DATE - last_purchase["Last_Purchase_Date"]
).dt.days
last_purchase["Churn"] = (last_purchase["Days_Since_Last_Purchase"] >= CHURN_DAYS).astype(int)

# ── RFM ───────────────────────────────────────────────────────────────────────
rfm = df.groupby("Customer_ID").agg(
    Recency   =("Invoice_Date", lambda x: (SNAPSHOT_DATE - x.max()).days),
    Frequency =("Transaction_ID","count"),
    Monetary  =("Total_Amount","sum"),
).reset_index()

# RFM Scores (1-5 quintiles, higher = better)
rfm["R_Score"] = pd.qcut(rfm["Recency"],  5, labels=[5,4,3,2,1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])
rfm["RFM_Score"]= rfm[["R_Score","F_Score","M_Score"]].astype(int).sum(axis=1)

# ── Customer metrics ──────────────────────────────────────────────────────────
metrics = df.groupby("Customer_ID").agg(
    Total_Revenue      =("Total_Amount","sum"),
    Avg_Order_Value    =("Total_Amount","mean"),
    Total_Orders       =("Transaction_ID","count"),
    Avg_Basket_Size    =("Quantity","mean"),
    Monthly_Spend      =("Total_Amount", lambda x: x.sum() / max(
                            (SNAPSHOT_DATE - df.loc[x.index,"Invoice_Date"].min()).days / 30, 1)),
).reset_index()

# CLV = AOV × Purchase Frequency × 12 months (simplified)
metrics["CLV"] = metrics["Avg_Order_Value"] * (metrics["Total_Orders"] / 12)

# Repeat purchase rate
order_counts = df.groupby("Customer_ID")["Invoice_Date"].nunique()
metrics["Repeat_Purchase_Rate"] = (
    order_counts.apply(lambda x: 1 if x > 1 else 0)
).values

# ── Behavioural features ──────────────────────────────────────────────────────
# Preferred category
pref_cat = (
    df.groupby(["Customer_ID","Category"])["Total_Amount"]
    .sum().reset_index()
    .sort_values("Total_Amount", ascending=False)
    .drop_duplicates("Customer_ID")[["Customer_ID","Category"]]
    .rename(columns={"Category":"Preferred_Category"})
)

# Weekend shopping ratio
df["DayOfWeek"] = df["Invoice_Date"].dt.dayofweek
weekend = df.groupby("Customer_ID").apply(
    lambda x: (x["DayOfWeek"] >= 5).sum() / len(x)
).reset_index()
weekend.columns = ["Customer_ID","Weekend_Ratio"]

# Payment preference
pay_pref = (
    df.groupby(["Customer_ID","Payment_Method"])["Transaction_ID"]
    .count().reset_index()
    .sort_values("Transaction_ID", ascending=False)
    .drop_duplicates("Customer_ID")[["Customer_ID","Payment_Method"]]
    .rename(columns={"Payment_Method":"Preferred_Payment"})
)

# Quarterly spend (last 90 days)
q_spend = df[df["Invoice_Date"] >= SNAPSHOT_DATE - pd.Timedelta(days=90)].groupby(
    "Customer_ID")["Total_Amount"].sum().reset_index()
q_spend.columns = ["Customer_ID","Quarterly_Spend"]

# ── Static customer attributes ────────────────────────────────────────────────
static = df.groupby("Customer_ID").agg(
    Customer_Age  =("Customer_Age","median"),
    Gender        =("Gender",lambda x: x.mode()[0] if len(x)>0 else "Unknown"),
    Country       =("Country",  lambda x: x.mode()[0]),
    Loyalty_Score =("Loyalty_Score","median"),
).reset_index()

# ── Merge all ─────────────────────────────────────────────────────────────────
feat = (last_purchase
        .merge(rfm,        on="Customer_ID")
        .merge(metrics,    on="Customer_ID")
        .merge(pref_cat,   on="Customer_ID", how="left")
        .merge(weekend,    on="Customer_ID", how="left")
        .merge(pay_pref,   on="Customer_ID", how="left")
        .merge(q_spend,    on="Customer_ID", how="left")
        .merge(static,     on="Customer_ID", how="left")
       )

feat["Quarterly_Spend"].fillna(0, inplace=True)

# Encode categoricals
feat = pd.get_dummies(feat, columns=["Gender","Country","Preferred_Category","Preferred_Payment"],
                      drop_first=False)

feat.to_csv(FEATURES, index=False)
print(f"Feature table saved → {feat.shape}  (rows=customers, cols=features)")
print(f"Churn rate          : {feat['Churn'].mean():.2%}")
