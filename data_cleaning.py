"""
data_cleaning.py
Cleans the raw retail transaction dataset.
Steps:
  1. Load raw data
  2. Remove duplicates
  3. Fix data types
  4. Handle missing values
  5. Remove invalid transactions (negative qty / zero price)
  6. Outlier capping
  7. Save cleaned dataset
"""

import pandas as pd
import numpy as np

RAW  = "/home/claude/Retail_Churn_Project/data/retail_transactions.csv"
CLEAN= "/home/claude/Retail_Churn_Project/data/cleaned_transactions.csv"

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW, parse_dates=["Invoice_Date","Last_Purchase_Date"])
print(f"Raw shape          : {df.shape}")

# ── 2. Duplicates ────────────────────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed : {before - len(df)}")

# ── 3. Data types ────────────────────────────────────────────────────────────
df["Customer_Age"]   = pd.to_numeric(df["Customer_Age"],   errors="coerce")
df["Loyalty_Score"]  = pd.to_numeric(df["Loyalty_Score"],  errors="coerce")
df["Total_Amount"]   = pd.to_numeric(df["Total_Amount"],   errors="coerce")
df["Quantity"]       = pd.to_numeric(df["Quantity"],        errors="coerce")

# ── 4. Missing values ────────────────────────────────────────────────────────
print(f"\nMissing before fill:\n{df.isnull().sum()[df.isnull().sum()>0]}")

df["Customer_Age"].fillna(df["Customer_Age"].median(), inplace=True)
df["Loyalty_Score"].fillna(df["Loyalty_Score"].median(), inplace=True)
df["Category"].fillna("Unknown", inplace=True)
df.dropna(subset=["Customer_ID","Transaction_ID","Invoice_Date"], inplace=True)

print(f"\nMissing after fill : {df.isnull().sum().sum()}")

# ── 5. Invalid transactions ──────────────────────────────────────────────────
before = len(df)
df = df[df["Quantity"] > 0]
df = df[df["Unit_Price"] > 0]
df = df[df["Total_Amount"] > 0]
# Recalculate Total_Amount to ensure consistency
df["Total_Amount"] = df["Quantity"] * df["Unit_Price"]
print(f"Invalid rows removed: {before - len(df)}")

# ── 6. Outlier capping (IQR) ─────────────────────────────────────────────────
for col in ["Total_Amount","Unit_Price","Quantity"]:
    Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(Q1, Q3)

# ── 7. Age sanity check ──────────────────────────────────────────────────────
df = df[(df["Customer_Age"] >= 18) & (df["Customer_Age"] <= 80)]

# ── 8. Save ──────────────────────────────────────────────────────────────────
df.to_csv(CLEAN, index=False)
print(f"\nCleaned shape      : {df.shape}")
print(f"Saved → {CLEAN}")
