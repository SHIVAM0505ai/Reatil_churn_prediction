# 🛒 Retail E-Commerce Customer Churn Prediction

**End-to-end ML project** — predicts which customers will stop purchasing in the next 90 days.

---

## 📁 Folder Structure

```
Retail_Churn_Project/
├── data/
│   ├── retail_transactions.csv     ← raw synthetic dataset
│   ├── cleaned_transactions.csv    ← after cleaning
│   └── customer_features.csv       ← RFM + ML features
├── src/
│   ├── generate_data.py            ← creates synthetic dataset
│   ├── data_cleaning.py            ← cleaning pipeline
│   ├── feature_engineering.py      ← RFM + behavioural features
│   ├── eda.py                      ← 10 EDA charts
│   ├── segmentation.py             ← K-Means clustering
│   └── ml_pipeline.py              ← model training + evaluation
├── models/
│   ├── best_churn_model.pkl        ← saved best model
│   └── feature_columns.pkl         ← feature list
├── reports/                        ← all charts saved here
├── app.py                          ← Streamlit web app
├── run_all.py                      ← runs full pipeline
└── requirements.txt
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python run_all.py

# 3. Launch Streamlit app
streamlit run app.py
```

---

## 🎯 Churn Definition
A customer is **churned** if they made **no purchase in the last 90 days**.

---

## 📊 Models Trained
| Model               | Notes                        |
|---------------------|------------------------------|
| Logistic Regression | Baseline, interpretable      |
| Decision Tree       | Explainable rules            |
| Random Forest       | Best overall performance     |
| Gradient Boosting   | Strong on imbalanced data    |

---

## 📈 Features Used
- **RFM**: Recency, Frequency, Monetary, RFM Score
- **Customer Metrics**: CLV, AOV, Repeat Purchase Rate
- **Behavioural**: Weekend Ratio, Preferred Category, Payment Preference
- **Demographics**: Age, Gender, Country, Loyalty Score

---

## 💡 Business Value
- Identify at-risk customers **before** they leave
- Target retention campaigns to high-risk segments
- Estimate **revenue saved** through proactive intervention
