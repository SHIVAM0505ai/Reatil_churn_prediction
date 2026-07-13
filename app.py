"""
app.py  —  Retail Customer Churn Prediction  (Streamlit)
Run:  streamlit run app.py
"""

import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import joblib, io, os # type: ignore

MODEL_PATH   = "models/best_churn_model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Churn Predictor")
st.sidebar.markdown("Upload customer CSV → get churn predictions instantly.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Retail E-Commerce — Customer Churn Prediction")
st.markdown("**Upload your customer feature file** to predict which customers are at risk of churning.")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

try:
    model, feature_cols = load_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model not found. Run ml_pipeline.py first. Error: {e}")
    st.stop()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Customer Feature CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Align columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)

    # Predict
    probs  = model.predict_proba(X)[:,1]
    preds  = model.predict(X)
    labels = ["Active" if p==0 else "Churned" for p in preds]
    risks  = ["🔴 High" if p>=0.7 else ("🟠 Medium" if p>=0.4 else "🟢 Low") for p in probs]

    df["Churn_Prediction"]   = labels
    df["Churn_Probability"]  = np.round(probs*100, 1)
    df["Risk_Level"]         = risks

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Customers",  len(df))
    col2.metric("Predicted Churned", int(preds.sum()))
    col3.metric("Retention Rate",   f"{(1-preds.mean())*100:.1f}%")
    col4.metric("Avg Churn Prob",   f"{probs.mean()*100:.1f}%")

    # ── Results Table ─────────────────────────────────────────────────────────
    st.subheader("🔍 Prediction Results")
    display_cols = (["Customer_ID"] if "Customer_ID" in df.columns else []) + \
                   ["Churn_Prediction","Churn_Probability","Risk_Level"]
    st.dataframe(df[display_cols], use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,4))
        pd.Series(labels).value_counts().plot(kind="bar", color=["#2E75B6","#E74C3C"],
                                               edgecolor="white", ax=ax)
        ax.set_title("Churn vs Active")
        ax.set_xticklabels(["Active","Churned"], rotation=0)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(probs*100, bins=30, color="#8E44AD", edgecolor="white", alpha=0.85)
        ax.set_title("Churn Probability Distribution")
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)

    # ── Business Recommendations ──────────────────────────────────────────────
    st.subheader("💡 Business Recommendations")
    high_risk = int((probs >= 0.7).sum())
    st.markdown(f"""
| Risk Level | Count | Recommended Action |
|---|---|---|
| 🔴 High (≥70%)  | {int((probs>=0.7).sum())} | Immediate retention campaign — offer personalised discount |
| 🟠 Medium (40-70%) | {int(((probs>=0.4)&(probs<0.7)).sum())} | Loyalty program nudge — email/push re-engagement |
| 🟢 Low (<40%)   | {int((probs<0.4).sum())} | Upsell / cross-sell campaigns |

**Estimated Revenue at Risk:** If avg CLV = ₹5,000 → **₹{high_risk*5000:,}** at immediate risk
    """)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Prediction Results", csv_out,
                        "churn_predictions.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file with customer features to get started.")
    st.markdown("""
### Expected CSV columns (numeric features):
`Recency, Frequency, Monetary, RFM_Score, Avg_Order_Value, CLV,
Days_Since_Last_Purchase, Loyalty_Score, Weekend_Ratio, Quarterly_Spend ...`

Run `feature_engineering.py` to generate this file from raw transactions.
    """)
