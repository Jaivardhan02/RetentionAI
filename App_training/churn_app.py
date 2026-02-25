"""
╔══════════════════════════════════════════════════════════════╗
║        RetentionAI — Customer Churn Predictor      ║
║        SELF-CONTAINED: Auto-trains model on first run       ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
    1. Put this file in the SAME FOLDER as your CSV files:
         customer_churn_dataset-training-master.csv
         customer_churn_dataset-testing-master.csv

    2. Install requirements (once):
         pip install streamlit scikit-learn pandas numpy matplotlib joblib

    3. Launch:
         streamlit run churn_app.py

NOTE ON .joblib FILES:
    .joblib files are BINARY — they look empty in a text editor.
    That is completely normal. The model is 23MB of compressed
    decision trees. This app trains and saves them automatically
    on first launch (~30 seconds). After that, they load instantly.
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PATHS  — everything lives next to this script
# ─────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE, "D:\Churn\Dataset\customer_churn_dataset-training-master.csv.zip")
TEST_CSV  = os.path.join(BASE, "D:\Churn\Dataset\customer_churn_dataset-testing-master.csv.zip")

MODEL_F   = os.path.join(BASE, "churn_model.joblib")
SCALER_F  = os.path.join(BASE, "scaler.joblib")          # ← NEW: StandardScaler
LE_G      = os.path.join(BASE, "le_gender.joblib")
LE_S      = os.path.join(BASE, "le_subscription.joblib")
LE_C      = os.path.join(BASE, "le_contract.joblib")
FEAT_F    = os.path.join(BASE, "feature_names.joblib")

ALL_ARTIFACTS = [MODEL_F, SCALER_F, LE_G, LE_S, LE_C, FEAT_F]

# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="RetentionAI | Churn Predictor",
                   page_icon="🚀", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{background:#0a0e1a;color:#e0e6f0;font-family:'Segoe UI',sans-serif}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1e2d4a}
.card{background:linear-gradient(135deg,#111827,#1a2540);border:1px solid #1e3a5f;
      border-radius:12px;padding:20px 24px;margin-bottom:16px;box-shadow:0 4px 20px rgba(0,0,0,.4)}
.risk-high{color:#ff4d6d;font-size:2rem;font-weight:800}
.risk-med{color:#ffd166;font-size:2rem;font-weight:800}
.risk-low{color:#06d6a0;font-size:2rem;font-weight:800}
.stButton>button{background:linear-gradient(90deg,#0066ff,#0044cc);color:white;border:none;
  border-radius:8px;padding:10px 28px;font-weight:600;width:100%;transition:all .2s}
.stButton>button:hover{background:linear-gradient(90deg,#0080ff,#0055ee);transform:translateY(-1px)}
.stTabs [data-baseweb="tab-list"]{background:#0f1629;border-radius:10px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#7a8fa6;font-weight:600}
.stTabs [aria-selected="true"]{background:#0066ff !important;color:white !important}
.hdr{background:linear-gradient(90deg,#001f4d,#003080,#001f4d);border-radius:14px;
     padding:24px 32px;margin-bottom:24px;border:1px solid #0047b3;display:flex;align-items:center;gap:16px}
.div{border:none;border-top:1px solid #1e3a5f;margin:16px 0}
.ah{background:rgba(255,77,109,.1);border:1px solid #ff4d6d;border-radius:8px;padding:12px 16px}
.al{background:rgba(6,214,160,.1);border:1px solid #06d6a0;border-radius:8px;padding:12px 16px}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  AUTO-TRAIN FUNCTION
#  Matches Churn_Prediction_Final.ipynb exactly:
#    • Both CSVs combined  (fixes distribution mismatch)
#    • 3-way split 70 / 15 / 15
#    • StandardScaler fit on train only  (saved as scaler.joblib)
#    • Fixed ordinal encoding (not random LabelEncoder order)
# ═════════════════════════════════════════════════════════════
def train_and_save(prog=None, status=None):
    def log(msg, p=None):
        if status: status.markdown(f"⚙️ {msg}")
        if prog and p: prog.progress(p)

    # ── 1. Load & combine both CSVs
    log("Loading and combining training + testing CSVs...", 0.05)
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV) if os.path.exists(TEST_CSV) else pd.DataFrame()
    df    = pd.concat([df_tr, df_te], axis=0, ignore_index=True) if len(df_te) else df_tr

    # ── 2. Drop leakage column + NaNs
    log("Removing CustomerID (data leakage) and NaN rows...", 0.15)
    df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
    df.dropna(inplace=True)

    # ── 3. Fixed ordinal encoding — same mapping every single run
    log("Encoding categoricals with fixed ordinal mapping...", 0.25)
    df["Gender"]            = df["Gender"].map({"Male": 0, "Female": 1})
    df["Subscription Type"] = df["Subscription Type"].map({"Basic": 0, "Standard": 1, "Premium": 2})
    df["Contract Length"]   = df["Contract Length"].map({"Monthly": 0, "Quarterly": 1, "Annual": 2})
    df.dropna(inplace=True)   # drop rows with unmapped values

    # Build LabelEncoders purely for the app's inverse-display (not for encoding)
    le_g = LabelEncoder().fit(["Male", "Female"])
    le_s = LabelEncoder().fit(["Basic", "Standard", "Premium"])
    le_c = LabelEncoder().fit(["Monthly", "Quarterly", "Annual"])

    # ── 4. 3-way stratified split: 70% train | 15% val | 15% test
    log("Splitting data: 70% train / 15% val / 15% test...", 0.35)
    X = df.drop("Churn", axis=1); y = df["Churn"].astype(int)
    feats = list(X.columns)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # ── 5. StandardScaler — fit ONLY on training data
    log("Fitting StandardScaler on training data only...", 0.45)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # fit + transform
    X_val_s   = scaler.transform(X_val)        # transform only — no leakage
    X_test_s  = scaler.transform(X_test)       # transform only — no leakage

    # ── 6. Train Random Forest
    log("Training Random Forest (200 trees, max_depth=15) — ~45 seconds...", 0.55)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,       # prevents overfitting
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train_s, y_train)

    # ── 7. Evaluate on val and test
    log("Evaluating on validation and test sets...", 0.82)
    val_auc  = roc_auc_score(y_val,  rf.predict_proba(X_val_s)[:,1])
    test_auc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:,1])
    val_acc  = accuracy_score(y_val,  rf.predict(X_val_s))
    test_acc = accuracy_score(y_test, rf.predict(X_test_s))
    overfit  = accuracy_score(y_train, rf.predict(X_train_s)) - val_acc

    # ── 8. Save all artifacts
    log("Saving model, scaler, and encoder files...", 0.92)
    joblib.dump(rf,     MODEL_F)
    joblib.dump(scaler, SCALER_F)   # ← CRITICAL: must be loaded before every prediction
    joblib.dump(le_g,   LE_G)
    joblib.dump(le_s,   LE_S)
    joblib.dump(le_c,   LE_C)
    joblib.dump(feats,  FEAT_F)

    log("Done!", 1.0)
    return val_auc, test_auc, val_acc, test_acc, overfit


# ═════════════════════════════════════════════════════════════
#  LOAD OR AUTO-TRAIN  (cached so it only runs once per session)
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    needs_train = any(not os.path.exists(f) for f in ALL_ARTIFACTS)

    if needs_train:
        if not os.path.exists(TRAIN_CSV):
            return None, None, None, None, None, None, "no_csv"

        st.markdown("### ⚙️ First-Time Setup: Training the Model")
        st.markdown(
            "No saved model found. Training now on your dataset — "
            "takes **~45 seconds**, happens **only once**."
        )
        pb     = st.progress(0.0)
        st_txt = st.empty()
        val_auc, test_auc, val_acc, test_acc, overfit = train_and_save(pb, st_txt)
        st_txt.success(
            f"✅ Training complete!  "
            f"Val AUC: **{val_auc:.4f}** | Test AUC: **{test_auc:.4f}** | "
            f"Val Acc: **{val_acc*100:.1f}%** | Test Acc: **{test_acc*100:.1f}%** | "
            f"Overfit gap: **{overfit:.4f}**"
        )
        pb.progress(1.0)
        st.info("Reloading with the trained model…")
        st.rerun()

    # ── Load all saved artifacts from disk
    rf     = joblib.load(MODEL_F)
    scaler = joblib.load(SCALER_F)   # ← load the scaler
    le_g   = joblib.load(LE_G)
    le_s   = joblib.load(LE_S)
    le_c   = joblib.load(LE_C)
    feats  = joblib.load(FEAT_F)
    return rf, scaler, le_g, le_s, le_c, feats, "ok"


model, scaler, le_g, le_s, le_c, FEATS, STATUS = get_model()

if STATUS == "no_csv":
    st.error(
        f"**Training CSV not found:**\n\n`{TRAIN_CSV}`\n\n"
        "Please place `customer_churn_dataset-training-master.csv` "
        "in the same folder as `churn_app.py` and restart."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

# Fixed ordinal maps — must match train_model.py exactly
_MAP_GENDER   = {"Male": 0, "Female": 1}
_MAP_SUB      = {"Basic": 0, "Standard": 1, "Premium": 2}
_MAP_CONTRACT = {"Monthly": 0, "Quarterly": 1, "Annual": 2}

def encode_row(age, gender, tenure, usage, support, delay, sub, contract, spend, last):
    """
    Encode one customer's inputs into a model-ready, SCALED DataFrame.
    Steps:
      1. Apply fixed ordinal maps (same as training)
      2. Build a raw DataFrame with correct column order
      3. Apply scaler.transform()  ← this is the key step that was missing
    """
    raw = pd.DataFrame([[
        age,
        _MAP_GENDER[gender],
        tenure,
        usage,
        support,
        delay,
        _MAP_SUB[sub],
        _MAP_CONTRACT[contract],
        spend,
        last
    ]], columns=FEATS)

    # ── Scale BEFORE passing to model (prevents prediction errors)
    raw_scaled = scaler.transform(raw)
    return raw_scaled   # returns a numpy array ready for model.predict_proba()

def risk(p):
    if p >= 0.65: return "HIGH RISK 🔴",   "risk-high"
    if p >= 0.35: return "MEDIUM RISK 🟡", "risk-med"
    return "LOW RISK 🟢", "risk-low"

def gauge(prob):
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0a0e1a")
    t = np.linspace(np.pi, 0, 300); r = 0.8
    ax.plot(r*np.cos(t), r*np.sin(t), lw=20, color="#1e3a5f", solid_capstyle="butt")
    t2 = np.linspace(np.pi, np.pi - prob*np.pi, 300)
    col = "#ff4d6d" if prob>=0.65 else "#ffd166" if prob>=0.35 else "#06d6a0"
    ax.plot(r*np.cos(t2), r*np.sin(t2), lw=20, color=col, solid_capstyle="butt")
    a = np.pi - prob*np.pi
    ax.annotate("", xy=(0.65*np.cos(a), 0.65*np.sin(a)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
    ax.text(0,-0.14,f"{prob*100:.1f}%",ha="center",va="center",
            fontsize=22,fontweight="bold",color="white")
    ax.text(0,-0.42,"Churn Probability",ha="center",va="center",fontsize=9,color="#7a8fa6")
    for lbl,xp in [("0%",-0.95),("50%",0),("100%",0.95)]:
        ax.text(xp,-0.05,lbl,ha="center",va="top",fontsize=7,color="#7a8fa6")
    ax.set_xlim(-1.15,1.15); ax.set_ylim(-0.6,1.1); ax.axis("off")
    plt.tight_layout(pad=0); return fig

def fi_chart():
    fi = pd.Series(model.feature_importances_, index=FEATS).sort_values()
    fig, ax = plt.subplots(figsize=(5,3.5))
    fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#111827")
    cols = ["#06d6a0" if v==fi.max() else "#0066ff" for v in fi.values]
    ax.barh(fi.index, fi.values, color=cols, edgecolor="none", height=0.65)
    ax.set_xlabel("Importance Score",color="#7a8fa6",fontsize=9)
    ax.tick_params(colors="#c0cfe0",labelsize=8)
    ax.set_title("Feature Importance",color="#c0cfe0",fontsize=10,pad=8)
    for s in ax.spines.values(): s.set_color("#1e3a5f")
    plt.tight_layout(); return fig

def batch(df_up):
    needed = ["Age","Gender","Tenure","Usage Frequency","Support Calls",
              "Payment Delay","Subscription Type","Contract Length",
              "Total Spend","Last Interaction"]
    miss = [c for c in needed if c not in df_up.columns]
    if miss: return None, f"Missing columns: {miss}"

    r = df_up.copy().dropna(subset=needed)
    try:
        # Apply same fixed ordinal encoding as training
        r["_g"] = r["Gender"].map(_MAP_GENDER)
        r["_s"] = r["Subscription Type"].map(_MAP_SUB)
        r["_c"] = r["Contract Length"].map(_MAP_CONTRACT)

        # Check for unmapped values (e.g. typos in uploaded CSV)
        if r[["_g","_s","_c"]].isnull().any().any():
            bad = r[r[["_g","_s","_c"]].isnull().any(axis=1)].index.tolist()
            return None, (f"Rows {bad[:5]} have unknown Gender/Subscription/Contract values. "
                          "Check spelling — expected: Female/Male, Basic/Standard/Premium, "
                          "Annual/Monthly/Quarterly.")

        X_raw = r[["Age","_g","Tenure","Usage Frequency","Support Calls",
                   "Payment Delay","_s","_c","Total Spend","Last Interaction"]].copy()
        X_raw.columns = FEATS

        # ── Scale BEFORE predicting (same scaler fitted during training)
        X_scaled = scaler.transform(X_raw)

        p = model.predict_proba(X_scaled)[:,1]
        r["Churn Probability (%)"] = (p*100).round(1)
        r["Risk Level"] = pd.cut(p, bins=[0,.35,.65,1.0],
                                   labels=["Low 🟢","Medium 🟡","High 🔴"])
        r.drop(columns=["_g","_s","_c"], inplace=True)
        return r.sort_values("Churn Probability (%)", ascending=False), None
    except Exception as e:
        return None, str(e)


# ═════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
        <span style='font-size:2.5rem'>🚀</span><br>
        <span style='font-size:1.3rem;font-weight:700;color:#e0e6f0'>RetentionAI</span><br>
        <span style='font-size:.75rem;color:#7a8fa6;letter-spacing:2px'>CHURN INTELLIGENCE</span>
    </div>""", unsafe_allow_html=True)

    st.success("✅ Model Ready")
    st.markdown(f"""
    <div class='card' style='padding:12px'>
    <div style='color:#7a8fa6;font-size:.75rem'>MODEL</div>
    <div style='color:#e0e6f0;font-weight:600'>Random Forest (200 trees)</div>
    <div style='color:#7a8fa6;font-size:.75rem;margin-top:8px'>SPLIT</div>
    <div style='color:#e0e6f0;font-weight:600'>70% Train / 15% Val / 15% Test</div>
    <div style='color:#7a8fa6;font-size:.75rem;margin-top:8px'>FEATURES</div>
    <div style='color:#e0e6f0;font-weight:600'>{len(FEATS)} (scaled before predict)</div>
    <div style='color:#7a8fa6;font-size:.75rem;margin-top:8px'>SAVED ARTIFACTS</div>
    <div style='color:#06d6a0;font-weight:600;font-size:.8rem'>
    ✓ churn_model.joblib<br>
    ✓ scaler.joblib ← applied before every predict<br>
    ✓ le_gender.joblib<br>
    ✓ le_subscription.joblib<br>
    ✓ le_contract.joblib<br>
    ✓ feature_names.joblib
    </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Retrain Model from Scratch"):
        for f in ALL_ARTIFACTS:   # ALL_ARTIFACTS now includes scaler.joblib
            if os.path.exists(f): os.remove(f)
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:#7a8fa6;font-size:.75rem'>
    <b style='color:#c0cfe0'>Top Churn Drivers</b><br><br>
    🔴 Support Calls (corr 0.51)<br>
    🟠 Monthly Contract (~62% churn)<br>
    🟡 Payment Delay ≥ 15 days<br>
    🟢 High Spend = more sticky<br>
    🔵 Long Tenure = lower risk
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("RetentionAI v1.0")


# ═════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class='hdr'>
  <span style='font-size:2.5rem'>🚀</span>
  <div>
    <div style='font-size:1.6rem;font-weight:800;color:#e0e6f0'>
      Customer Churn Intelligence Dashboard</div>
    <div style='color:#7a8fa6;font-size:.9rem;margin-top:2px'>
      RetentionAI &nbsp;•&nbsp; Random Forest AI &nbsp;•&nbsp; Real-time risk scoring
    </div>
  </div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Individual Prediction", "📊 Risk Analysis",
    "📂 Batch Processing",      "📖 How It Works"])


# ─────────────────────────────────────────────────────────────
#  TAB 1 — INDIVIDUAL PREDICTION
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🎯 Predict Churn for a Single Customer")
    L, R = st.columns([1.2, 1], gap="large")

    with L:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**👤 Demographics**")
        c1, c2 = st.columns(2)
        with c1:
            age    = st.slider("Age", 18, 65, 35)
            gender = st.selectbox("Gender", ["Female", "Male"])
        with c2:
            tenure   = st.slider("Tenure (months)", 1, 60, 24)
            sub_type = st.selectbox("Subscription", ["Basic","Standard","Premium"])

        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("**📞 Engagement**")
        c3, c4 = st.columns(2)
        with c3:
            support_calls = st.slider("Support Calls", 0, 10, 3,
                                       help="⚠️ Strongest churn signal (corr: 0.51)")
            usage_freq    = st.slider("Usage Freq/month", 1, 30, 15)
        with c4:
            payment_delay    = st.slider("Payment Delay (days)", 0, 30, 5)
            last_interaction = st.slider("Days Since Last Interaction", 1, 30, 10)

        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("**💳 Billing**")
        c5, c6 = st.columns(2)
        with c5:
            total_spend = st.slider("Total Spend ($)", 100, 1000, 500)
        with c6:
            contract = st.selectbox("Contract", ["Annual","Monthly","Quarterly"],
                                     help="⚠️ Monthly → ~62% churn rate")
        st.markdown("</div>", unsafe_allow_html=True)
        btn = st.button("🔮 Predict Churn Risk", use_container_width=True)

    with R:
        if btn:
            row_scaled = encode_row(age, gender, tenure, usage_freq, support_calls,
                                    payment_delay, sub_type, contract, total_spend, last_interaction)
            prob = model.predict_proba(row_scaled)[0][1]   # row_scaled is already numpy array
            lbl, css = risk(prob)

            st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
            st.markdown(f"<div class='{css}'>{lbl}</div>", unsafe_allow_html=True)
            st.pyplot(gauge(prob), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**🔍 Risk Factors**")
            ins = []
            if support_calls >= 6: ins.append("🔴 **High Support Calls** — strongest churn signal")
            if payment_delay >= 15: ins.append("🟠 **High Payment Delay** — disengagement sign")
            if contract == "Monthly": ins.append("🟡 **Monthly Contract** — ~62% churn rate")
            if tenure <= 6: ins.append("🟡 **Short Tenure** — new customers churn more")
            if usage_freq <= 5: ins.append("🟠 **Low Usage** — losing interest")
            if total_spend >= 800: ins.append("🟢 **High Spender** — likely to stay")
            if sub_type == "Premium": ins.append("🟢 **Premium Sub** — higher loyalty")
            if not ins: ins.append("🟢 **No major risk flags** — profile looks stable")
            for i in ins: st.markdown(f"- {i}")
            st.markdown("</div>", unsafe_allow_html=True)

            if prob >= 0.65:
                st.markdown("""<div class='ah'><b>⚡ Immediate Actions</b><br>
                • Assign dedicated CSM<br>• Offer annual contract discount<br>
                • Win-back campaign within 48hrs</div>""", unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown("""<div class='card'><b>💡 Suggested Actions</b><br>
                • Schedule check-in call<br>• Send usage tips<br>
                • Consider loyalty reward</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='al'><b>✅ Customer is Stable</b><br>
                • Standard engagement<br>• Consider Premium upsell</div>""",
                            unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='text-align:center;padding:60px 20px'>
                <span style='font-size:3rem'>🔮</span><br><br>
                <span style='color:#7a8fa6'>Fill the profile on the left and click<br>
                <b style='color:#0066ff'>Predict Churn Risk</b></span>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 2 — RISK ANALYSIS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 Model & Risk Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**🎯 Feature Importance**")
        st.pyplot(fi_chart(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**📋 Risk Tiers**")
        st.markdown("""
| Level | Probability | Action |
|---|---|---|
| 🟢 Low | 0–35% | Standard |
| 🟡 Medium | 35–65% | Proactive |
| 🔴 High | 65–100% | Immediate |
        """)
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        st.markdown("**🧠 Model Info**")
        st.markdown("""
- **Algorithm:** Random Forest (100 trees)  
- **Training rows:** ~440K  
- **Validation AUC:** ~1.00  
- **Strongest feature:** Support Calls  
- **Leakage removed:** CustomerID dropped  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**📉 Churn Rate by Contract Type**")
    fig2, ax2 = plt.subplots(figsize=(8,2.5))
    fig2.patch.set_facecolor("#0a0e1a"); ax2.set_facecolor("#111827")
    bars = ax2.bar(["Annual","Quarterly","Monthly"],[10,28,62],
                   color=["#06d6a0","#ffd166","#ff4d6d"],width=0.5,edgecolor="none")
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f"{b.get_height()}%", ha="center",va="bottom",
                 color="white",fontsize=11,fontweight="bold")
    ax2.set_ylabel("Churn Rate %",color="#7a8fa6",fontsize=9)
    ax2.set_ylim(0,80); ax2.tick_params(colors="#c0cfe0")
    for s in ax2.spines.values(): s.set_color("#1e3a5f")
    plt.tight_layout(); st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 3 — BATCH PROCESSING
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📂 Batch Churn Prediction")
    st.markdown("""
    <div class='card'>
    <b>Required CSV columns:</b><br>
    <code>Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay,
    Subscription Type, Contract Length, Total Spend, Last Interaction</code><br><br>
    <b>Gender:</b> Female / Male &nbsp;|&nbsp;
    <b>Subscription Type:</b> Basic / Standard / Premium &nbsp;|&nbsp;
    <b>Contract Length:</b> Annual / Monthly / Quarterly
    </div>""", unsafe_allow_html=True)

    if os.path.exists(TEST_CSV):
        st.info(f"💡 Your test CSV is ready: `{TEST_CSV}` — upload it below to try!")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df_up = pd.read_csv(up)
        st.markdown(f"**Loaded {len(df_up):,} records.** Preview:")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Prediction", use_container_width=True):
            with st.spinner("Predicting..."):
                res, err = batch(df_up)
            if err:
                st.error(f"Error: {err}")
            else:
                st.success(f"✅ Done — {len(res):,} customers predicted!")
                hi = (res["Risk Level"]=="High 🔴").sum()
                me = (res["Risk Level"]=="Medium 🟡").sum()
                lo = (res["Risk Level"]=="Low 🟢").sum()
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total",f"{len(res):,}")
                m2.metric("🔴 High", f"{hi:,}", delta=f"{hi/len(res)*100:.1f}%", delta_color="inverse")
                m3.metric("🟡 Medium",f"{me:,}", delta=f"{me/len(res)*100:.1f}%", delta_color="off")
                m4.metric("🟢 Low",   f"{lo:,}", delta=f"{lo/len(res)*100:.1f}%")
                show = [c for c in ["CustomerID","Age","Gender","Contract Length",
                                    "Support Calls","Churn Probability (%)","Risk Level"]
                        if c in res.columns]
                st.dataframe(res[show].head(200), use_container_width=True)
                st.download_button("⬇️ Download Predictions CSV",
                                   data=res.to_csv(index=False).encode(),
                                   file_name="churn_predictions.csv",
                                   mime="text/csv", use_container_width=True)
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:50px 20px'>
            <span style='font-size:3rem'>📂</span><br><br>
            <span style='color:#7a8fa6'>Upload a CSV above.<br>
            Try <code>customer_churn_dataset-testing-master.csv</code></span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 4 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 📖 Workflow Guide")

    st.markdown("""
    <div class='card'>
    <h4 style='color:#0080ff;margin-top:0'>🗺️ End-to-End Pipeline</h4>
    <pre style='background:#05080f;color:#c0cfe0;padding:16px;border-radius:8px;
                font-size:.82rem;border:1px solid #1e3a5f'>
  training.csv  ──▶  Auto-Training  ──▶  .joblib Files  ──▶  Streamlit UI
  (440K rows)        (first launch)      churn_model         Tab 1: Single predict
                                         le_gender            Tab 2: Risk analysis
                                         le_subscription      Tab 3: Batch predict
                                         le_contract          Tab 4: This guide
                                         feature_names
    </pre>
    </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("""
        <div class='card'>
        <h4 style='color:#0080ff;margin-top:0'>📁 File Layout</h4>
        <pre style='background:#05080f;color:#c0cfe0;padding:12px;border-radius:8px;font-size:.82rem;border:1px solid #1e3a5f'>
your-project/
├── churn_app.py            ← This file
├── training-master.csv     ← Training data
├── testing-master.csv      ← For batch upload
│
└── Auto-created on 1st run:
    ├── churn_model.joblib  (23 MB binary)
    ├── le_gender.joblib
    ├── le_subscription.joblib
    ├── le_contract.joblib
    └── feature_names.joblib
        </pre>
        </div>""", unsafe_allow_html=True)

    with cb:
        st.markdown("""
        <div class='card'>
        <h4 style='color:#0080ff;margin-top:0'>⚡ Commands</h4>
        <pre style='background:#05080f;color:#c0cfe0;padding:12px;border-radius:8px;font-size:.82rem;border:1px solid #1e3a5f'>
# Install (once)
pip install streamlit scikit-learn \\
            pandas numpy matplotlib joblib

# Run (trains on first launch automatically)
streamlit run churn_app.py
        </pre>
        <h4 style='color:#0080ff;margin-top:14px'>💾 Why .joblib Files Look Empty</h4>
        <p style='color:#c0cfe0;font-size:.88rem'>
        <code>.joblib</code> files are <b>binary serialized objects</b> —
        like a ZIP of your trained model. They look empty or garbled
        in text editors because they're not text. They're 100% normal
        and contain 23MB of compressed Random Forest decision trees.
        Just keep them in the same folder as <code>churn_app.py</code>.
        </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h4 style='color:#0080ff;margin-top:0'>🔬 EDA Findings Baked Into This App</h4>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
                color:#c0cfe0;font-size:.88rem'>
    <div><b style='color:#ff4d6d'>🔴 Highest Risk</b><br>
    • Support Calls ≥ 6<br>• Monthly contract<br>• Payment Delay ≥ 15d</div>
    <div><b style='color:#ffd166'>🟡 Moderate Risk</b><br>
    • Tenure ≤ 6 months<br>• Usage Freq ≤ 5/mo<br>• Quarterly contract</div>
    <div><b style='color:#06d6a0'>🟢 Retention Factors</b><br>
    • Annual contract (~10%)<br>• Premium subscription<br>• High spend customers</div>
    </div></div>""", unsafe_allow_html=True)