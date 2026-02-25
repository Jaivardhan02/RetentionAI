"""
╔══════════════════════════════════════════════════════════════════════════╗
║     RetentionAI — Model Training Script                        ║
║                                                                         ║
║     KEY FEATURES:                             ║
║       1. 3-WAY SPLIT  : 70% Train | 15% Val | 15% Test                 ║
║          (Previous version only did 80/20, no holdout test set)         ║
║       2. FEATURE SCALING : StandardScaler fit on train only             ║
║          (Prevents data leakage from val/test into the scaler)          ║
║       3. MULTI-MODEL  : Trains 5 models, picks the best by Val Acc      ║
║          (LR, SVM, Decision Tree, Random Forest, Gradient Boosting)     ║
║       4. SCALER SAVED : scaler.joblib saved alongside the model         ║
║          (Streamlit app must scale inputs the same way before predict)  ║
║       5. BOTH CSVs    : Train + Test CSVs combined before splitting     ║
║          (Fixes distribution mismatch between the two raw files)        ║
╚══════════════════════════════════════════════════════════════════════════╝

USAGE:
    python train_model.py

OUTPUT FILES  (saved in same folder as this script):
    churn_model.joblib        ← Best model (chosen by Validation Accuracy)
    scaler.joblib             ← StandardScaler (MUST be used in the app too)
    le_gender.joblib          ← LabelEncoder for Gender
    le_subscription.joblib    ← LabelEncoder for Subscription Type
    le_contract.joblib        ← LabelEncoder for Contract Length
    feature_names.joblib      ← Ordered list of feature column names
    model_report.txt          ← Full evaluation report saved to disk
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration      import CalibratedClassifierCV
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.svm              import LinearSVC
from sklearn.tree             import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  CONFIG  ← Edit these paths to match your folder structure
# ══════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# If your CSVs are zipped, pandas reads .zip directly.
# If they are plain .csv files, just remove the .zip extension below.
TRAIN_CSV = os.path.join(BASE_DIR, "D:\Churn\Dataset\customer_churn_dataset-training-master.csv.zip")
TEST_CSV  = os.path.join(BASE_DIR, "D:\Churn\Dataset\customer_churn_dataset-testing-master.csv.zip")

# Alternatively, if they are still .zip:
# TRAIN_CSV = os.path.join(BASE_DIR, "D:\Churn\Dataset\customer_churn_dataset-training-master.csv.zip")
# TEST_CSV  = os.path.join(BASE_DIR, "D:\Churn\Dataset\customer_churn_dataset-testing-master.csv.zip")

RANDOM_STATE = 42

# ══════════════════════════════════════════════════════════════
#  OUTPUT FILE PATHS  (all saved next to this script)
# ══════════════════════════════════════════════════════════════
MODEL_F   = os.path.join(BASE_DIR, "churn_model.joblib")
SCALER_F  = os.path.join(BASE_DIR, "scaler.joblib")
LE_G      = os.path.join(BASE_DIR, "le_gender.joblib")
LE_S      = os.path.join(BASE_DIR, "le_subscription.joblib")
LE_C      = os.path.join(BASE_DIR, "le_contract.joblib")
FEAT_F    = os.path.join(BASE_DIR, "feature_names.joblib")
REPORT_F  = os.path.join(BASE_DIR, "model_report.txt")

# ══════════════════════════════════════════════════════════════
#  HELPER — pretty section header
# ══════════════════════════════════════════════════════════════
def header(step, title):
    print(f"\n{'='*65}")
    print(f"  [{step}]  {title}")
    print(f"{'='*65}")

def ok(msg):
    print(f"      ✓  {msg}")

def info(msg):
    print(f"      ℹ  {msg}")

# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & COMBINE BOTH CSVs
# ══════════════════════════════════════════════════════════════
header("1/6", "LOAD DATA")

for path in [TRAIN_CSV, TEST_CSV]:
    if not os.path.exists(path):
        print(f"\n  ❌ File not found: {path}")
        print("     Please update TRAIN_CSV / TEST_CSV paths at the top of this script.")
        sys.exit(1)

df_train_raw = pd.read_csv(TRAIN_CSV)
df_test_raw  = pd.read_csv(TEST_CSV)

ok(f"Training CSV loaded  → {df_train_raw.shape[0]:,} rows × {df_train_raw.shape[1]} cols")
ok(f"Testing  CSV loaded  → {df_test_raw.shape[0]:,} rows × {df_test_raw.shape[1]} cols")

# Why we combine: the two raw CSVs have very different churn distributions.
# Combining and re-splitting gives a balanced, representative dataset.
info(f"Train CSV churn rate : {df_train_raw['Churn'].mean()*100:.1f}%")
info(f"Test  CSV churn rate : {df_test_raw['Churn'].mean()*100:.1f}%")
info("Distributions differ → combining both before splitting (matches notebook)")

df = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)
ok(f"Combined dataset     → {len(df):,} rows × {len(df.columns)} cols")
ok(f"Combined churn rate  → {df['Churn'].mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
#  STEP 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
header("2/6", "PREPROCESSING")

# 2a. Drop CustomerID — data leakage (corr ~0.51 in raw data)
df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
ok("Dropped CustomerID  (identified as data leakage in EDA)")

# 2b. Label-encode categoricals using fixed ordinal mapping
#     (matches the notebook's map_dict approach exactly)
map_gender   = {"Male": 0, "Female": 1}
map_sub      = {"Basic": 0, "Standard": 1, "Premium": 2}
map_contract = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

# Fit LabelEncoders so the Streamlit app can inverse-transform for display
le_gender_enc   = LabelEncoder().fit(list(map_gender.keys()))
le_sub_enc      = LabelEncoder().fit(list(map_sub.keys()))
le_contract_enc = LabelEncoder().fit(list(map_contract.keys()))

df["Gender"]            = df["Gender"].map(map_gender)
df["Subscription Type"] = df["Subscription Type"].map(map_sub)
df["Contract Length"]   = df["Contract Length"].map(map_contract)

ok(f"Encoded Gender           → {map_gender}")
ok(f"Encoded Subscription     → {map_sub}")
ok(f"Encoded Contract Length  → {map_contract}")

# 2c. Drop NaN rows (result of unmapped/unknown category values)
before = len(df)
df.dropna(inplace=True)
ok(f"Removed {before - len(df):,} rows with NaN values  ({len(df):,} rows remain)")

# ══════════════════════════════════════════════════════════════
#  STEP 3 — 3-WAY STRATIFIED SPLIT  (matches notebook exactly)
#           70% Train | 15% Validation | 15% Test
# ══════════════════════════════════════════════════════════════
header("3/6", "3-WAY STRATIFIED SPLIT  (70 / 15 / 15)")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)
FEATURES = list(X.columns)

# Step A: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# Step B: Split 30% temp into 50/50 → 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\n  {'Split':<14} {'Rows':>10}  {'Churn Rate':>12}")
print(f"  {'-'*40}")
print(f"  {'Train':<14} {len(X_train):>10,}  {y_train.mean():>11.1%}")
print(f"  {'Validation':<14} {len(X_val):>10,}  {y_val.mean():>11.1%}")
print(f"  {'Test':<14} {len(X_test):>10,}  {y_test.mean():>11.1%}")
print(f"  {'Total':<14} {len(X):>10,}  {y.mean():>11.1%}")

info("Stratify=True ensures equal churn ratio across all three splits")

# ══════════════════════════════════════════════════════════════
#  STEP 4 — FEATURE SCALING  (fit on train ONLY)
#           Prevents data leakage from val/test into scaler
# ══════════════════════════════════════════════════════════════
header("4/6", "FEATURE SCALING  (StandardScaler — fit on TRAIN only)")

scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train)   # Fit + transform
X_val_s    = scaler.transform(X_val)          # Transform only (no fit)
X_test_s   = scaler.transform(X_test)         # Transform only (no fit)

ok(f"Scaler fit on training set  → {X_train_s.shape}")
ok(f"Val  scaled (no re-fit)     → {X_val_s.shape}")
ok(f"Test scaled (no re-fit)     → {X_test_s.shape}")
info("Scaler will be saved and used by the Streamlit app for consistency")

# ══════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN ALL MODELS & COMPARE
# ══════════════════════════════════════════════════════════════
header("5/6", "TRAIN ALL MODELS  (LR / SVM / DT / RF / GB)")

model_zoo = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=RANDOM_STATE
    ),
    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(dual=False, C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, random_state=RANDOM_STATE   # max_depth prunes overfitting
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,          # Limits tree depth to reduce overfitting
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,           # Shallow trees = less overfit for boosting
        learning_rate=0.1,
        random_state=RANDOM_STATE
    ),
}

results       = []
trained_zoo   = {}

print(f"\n  {'Model':<24} {'Train Acc':>10} {'Val Acc':>10} "
      f"{'Test Acc':>10} {'Gap (Ovfit)':>12} {'Val AUC':>9}")
print(f"  {'-'*79}")

for name, m in model_zoo.items():
    # ── Train on training split only
    m.fit(X_train_s, y_train)
    trained_zoo[name] = m

    # ── Predict on all three splits
    tr_acc  = accuracy_score(y_train, m.predict(X_train_s))
    val_acc = accuracy_score(y_val,   m.predict(X_val_s))
    te_acc  = accuracy_score(y_test,  m.predict(X_test_s))
    gap     = tr_acc - val_acc

    # ── ROC-AUC on validation (used for model selection)
    try:
        val_auc = roc_auc_score(y_val, m.predict_proba(X_val_s)[:, 1])
    except Exception:
        val_auc = float("nan")

    results.append({
        "Model":       name,
        "Train Acc":   round(tr_acc,  4),
        "Val Acc":     round(val_acc, 4),
        "Test Acc":    round(te_acc,  4),
        "Overfit Gap": round(gap,     4),
        "Val AUC":     round(val_auc, 4),
    })

    # Flag if overfitting is noticeable
    flag = "  ⚠️  Overfitting!" if gap > 0.05 else ""
    print(f"  ✅ {name:<22} {tr_acc:>10.4f} {val_acc:>10.4f} "
          f"{te_acc:>10.4f} {gap:>12.4f} {val_auc:>9.4f}{flag}")

# ── Sort by Validation Accuracy (same criterion as notebook)
results_df   = pd.DataFrame(results).sort_values("Val Acc", ascending=False)
best_row     = results_df.iloc[0]
best_name    = best_row["Model"]
best_model   = trained_zoo[best_name]

print(f"\n  {'='*50}")
print(f"  🏆 BEST MODEL        : {best_name}")
print(f"     Validation Acc    : {best_row['Val Acc']:.4f}")
print(f"     Final Test Acc    : {best_row['Test Acc']:.4f}")
print(f"     Overfitting Gap   : {best_row['Overfit Gap']:.4f}  "
      f"(Train−Val; closer to 0 = better generalisation)")
print(f"     Validation AUC    : {best_row['Val AUC']:.4f}")
print(f"  {'='*50}")

# ══════════════════════════════════════════════════════════════
#  FULL CLASSIFICATION REPORTS  (Val + Test on best model)
# ══════════════════════════════════════════════════════════════
y_val_pred  = best_model.predict(X_val_s)
y_test_pred = best_model.predict(X_test_s)

val_report  = classification_report(y_val,  y_val_pred,  target_names=["No Churn","Churn"])
test_report = classification_report(y_test, y_test_pred, target_names=["No Churn","Churn"])

print(f"\n  --- {best_name}: VALIDATION SET ---")
print(val_report)

print(f"  --- {best_name}: FINAL TEST SET (unseen data) ---")
print(test_report)

# ── AUC gap between val and test (small gap = good generalisation)
y_prob_val  = best_model.predict_proba(X_val_s)[:, 1]
y_prob_test = best_model.predict_proba(X_test_s)[:, 1]
auc_val     = roc_auc_score(y_val,  y_prob_val)
auc_test    = roc_auc_score(y_test, y_prob_test)
print(f"  Validation AUC : {auc_val:.4f}")
print(f"  Test AUC       : {auc_test:.4f}")
print(f"  AUC Gap        : {auc_val - auc_test:.4f}  "
      f"(small gap = good generalisation ✅)")

# ── Feature importance (available only for tree-based models)
print(f"\n  Feature Importances ({best_name}):")
print(f"  {'-'*40}")
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, score in fi.items():
        bar = "█" * int(score * 40)
        print(f"  {feat:<22} {score:.4f}  {bar}")
else:
    info("Feature importances not available for this model type")

# ── Confusion matrices
cm_val  = confusion_matrix(y_val,  y_val_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print(f"\n  Confusion Matrix — Validation:")
print(f"  {cm_val}")
print(f"\n  Confusion Matrix — Test:")
print(f"  {cm_test}")

# ── Save full report to disk
report_lines = [
    "RetentionAI — Churn Model Training Report",
    "=" * 65,
    "",
    "MODEL COMPARISON TABLE",
    results_df.to_string(index=False),
    "",
    f"BEST MODEL: {best_name}",
    f"Validation Accuracy : {best_row['Val Acc']:.4f}",
    f"Test Accuracy       : {best_row['Test Acc']:.4f}",
    f"Overfitting Gap     : {best_row['Overfit Gap']:.4f}",
    f"Validation AUC      : {auc_val:.4f}",
    f"Test AUC            : {auc_test:.4f}",
    f"AUC Gap             : {auc_val - auc_test:.4f}",
    "",
    f"VALIDATION SET — Classification Report ({best_name})",
    val_report,
    "",
    f"TEST SET — Classification Report ({best_name})",
    test_report,
]
with open(REPORT_F, "w") as fp:
    fp.write("\n".join(report_lines))

# ══════════════════════════════════════════════════════════════
#  STEP 6 — SAVE ALL ARTIFACTS
# ══════════════════════════════════════════════════════════════
header("6/6", "SAVE ARTIFACTS")

joblib.dump(best_model,      MODEL_F)
joblib.dump(scaler,          SCALER_F)
joblib.dump(le_gender_enc,   LE_G)
joblib.dump(le_sub_enc,      LE_S)
joblib.dump(le_contract_enc, LE_C)
joblib.dump(FEATURES,        FEAT_F)

for label, path in [
    ("churn_model.joblib     ← Best model binary (NOT human-readable — this is normal)", MODEL_F),
    ("scaler.joblib          ← StandardScaler state",                                   SCALER_F),
    ("le_gender.joblib       ← Gender encoder",                                         LE_G),
    ("le_subscription.joblib ← Subscription encoder",                                   LE_S),
    ("le_contract.joblib     ← Contract encoder",                                       LE_C),
    ("feature_names.joblib   ← Feature column order",                                   FEAT_F),
    ("model_report.txt       ← Full evaluation report (human-readable)",                REPORT_F),
]:
    size = os.path.getsize(path)
    size_str = f"{size/1_000_000:.1f} MB" if size > 1_000_000 else f"{size/1_000:.1f} KB"
    ok(f"{label:<58}  [{size_str}]")

print(f"""
{'='*65}
  ✅  Training complete!

  NEXT STEP:
      streamlit run churn_app.py

{'='*65}
""")