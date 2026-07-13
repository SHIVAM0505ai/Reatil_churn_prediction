"""
run_all.py  —  Executes the full churn prediction pipeline in order.
"""
import subprocess, sys, time

steps = [
    ("1. Generate Data",        "src/generate_data.py"),
    ("2. Data Cleaning",        "src/data_cleaning.py"),
    ("3. Feature Engineering",  "src/feature_engineering.py"),
    ("4. EDA",                  "src/eda.py"),
    ("5. Segmentation",         "src/segmentation.py"),
    ("6. ML Pipeline",          "src/ml_pipeline.py"),
]

base = "/home/claude/Retail_Churn_Project"

for title, script in steps:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    t0 = time.time()
    result = subprocess.run([sys.executable, f"{base}/{script}"],
                             capture_output=False, cwd=base)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ❌  {title} FAILED")
        sys.exit(1)
    print(f"  ✅  Done in {elapsed:.1f}s")

print("\n" + "="*55)
print("  🎉  Full pipeline complete!")
print("  Reports saved to → Retail_Churn_Project/reports/")
print("  Model saved to   → Retail_Churn_Project/models/")
print("  Run app:  streamlit run app.py")
print("="*55)
