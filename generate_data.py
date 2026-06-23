"""
generate_data.py
Generates a realistic synthetic retail e-commerce dataset for churn modelling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

SNAPSHOT_DATE = datetime(2024, 12, 31)
N_CUSTOMERS   = 5000
N_TRANSACTIONS = 40000

CATEGORIES     = ["Electronics","Clothing","Home & Kitchen","Books","Sports","Beauty","Grocery","Toys"]
COUNTRIES      = ["India","USA","UK","Germany","France","Canada","Australia","UAE"]
PAYMENT_METHODS= ["Credit Card","Debit Card","UPI","Net Banking","Wallet","COD"]
GENDERS        = ["Male","Female","Other"]

# ── customer master ──────────────────────────────────────────────────────────
customers = pd.DataFrame({
    "Customer_ID"      : [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS+1)],
    "Customer_Age"     : np.random.randint(18, 70, N_CUSTOMERS),
    "Gender"           : np.random.choice(GENDERS, N_CUSTOMERS, p=[0.50,0.47,0.03]),
    "Country"          : np.random.choice(COUNTRIES, N_CUSTOMERS,
                            p=[0.35,0.20,0.12,0.08,0.07,0.07,0.06,0.05]),
    "Loyalty_Score"    : np.round(np.random.beta(2,3,N_CUSTOMERS)*10, 1),
})

# ── transactions ─────────────────────────────────────────────────────────────
# Churn-prone customers buy less recently; active customers buy more recently
churn_flag = np.random.choice([0,1], N_CUSTOMERS, p=[0.65,0.35])

rows = []
for idx, row in customers.iterrows():
    is_churned = churn_flag[idx]
    n_orders = np.random.randint(1,6) if is_churned else np.random.randint(3,25)

    for _ in range(n_orders):
        if is_churned:
            days_ago = np.random.randint(90, 540)
        else:
            days_ago = np.random.randint(0, 89)

        invoice_date = SNAPSHOT_DATE - timedelta(days=int(days_ago))
        qty          = np.random.randint(1, 10)
        unit_price   = round(np.random.uniform(5, 800), 2)

        rows.append({
            "Transaction_ID"    : f"T{len(rows)+1:07d}",
            "Customer_ID"       : row["Customer_ID"],
            "Invoice_Date"      : invoice_date,
            "Product_ID"        : f"P{np.random.randint(1,500):04d}",
            "Category"          : np.random.choice(CATEGORIES),
            "Quantity"          : qty,
            "Unit_Price"        : unit_price,
            "Total_Amount"      : round(qty * unit_price, 2),
            "Payment_Method"    : np.random.choice(PAYMENT_METHODS),
            "Purchase_Frequency": np.random.randint(1, 30),
        })

df_tx = pd.DataFrame(rows)

# Merge customer info
df = df_tx.merge(customers, on="Customer_ID")
df["Last_Purchase_Date"] = df.groupby("Customer_ID")["Invoice_Date"].transform("max")

# Inject 3% nulls in a few columns
for col in ["Customer_Age","Loyalty_Score","Category"]:
    mask = np.random.rand(len(df)) < 0.03
    df.loc[mask, col] = np.nan

# Inject ~1% duplicate rows
dup_idx = np.random.choice(df.index, int(len(df)*0.01), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# Inject negative quantities (~0.5%)
neg_idx = np.random.choice(df.index, int(len(df)*0.005), replace=False)
df.loc[neg_idx, "Quantity"] = -df.loc[neg_idx, "Quantity"]

df.to_csv("/home/claude/Retail_Churn_Project/data/retail_transactions.csv", index=False)
print(f"Dataset saved  →  {len(df):,} rows  |  {df['Customer_ID'].nunique():,} customers")
