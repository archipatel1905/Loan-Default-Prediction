import streamlit as st
import joblib
import sklearn
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BagAI · Loan Risk Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BAGGING MODEL  (prediction logic unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("bagging.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — track prediction history for live analytics
# ─────────────────────────────────────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "total_predictions" not in st.session_state:
    st.session_state.total_predictions = 0

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root { color-scheme: light; }
html, body, .stApp { background: #F3F6FB; color: #172B4D; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background: #F3F6FB !important; }
[data-testid="stMainBlockContainer"] { max-width: 1200px; padding: 1.5rem 2.5rem 4rem; }
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] { display:none; }
[data-testid="stHeader"] { background:transparent; }
h1,h2,h3,h4,h5,h6 { color:#172B4D !important; }
a { color:#315EFB; }
.topnav { display:flex; align-items:center; justify-content:space-between; gap:24px; padding:18px 0 24px; border-bottom:1px solid #DFE6F1; }
.nav-logo { display:flex; align-items:center; gap:10px; font-size:1.3rem; font-weight:800; }
.logo-text { color:#315EFB; }
.nav-links { display:flex; gap:24px; list-style:none; margin:0; padding:0; }
.nav-links a { color:#5E6E87; font-size:.85rem; text-decoration:none; }
.nav-links a:hover, .nav-links a.active { color:#315EFB; }
.nav-status { display:flex; align-items:center; gap:8px; font-size:.75rem; color:#315EFB; background:#EAF0FF; padding:8px 12px; border-radius:6px; }
.status-dot { width:6px; height:6px; border-radius:50%; background:#315EFB; }
.section-pill { display:inline-block; color:#315EFB; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; margin-bottom:12px; }
.shimmer-text { color:#315EFB; }
.glass-card, .stat-card { background:#FFFFFF !important; border:1px solid #DFE6F1 !important; border-radius:12px !important; box-shadow:0 3px 12px #172e2a04; }
.stat-card { text-align:left !important; padding:22px !important; }
.hero-card { background:#142B50; border:1px solid #243E64; border-radius:16px; padding:30px; margin-top:8px; }
.hero-card .eyebrow { font-size:.7rem; letter-spacing:1.6px; font-weight:700; color:#B7C7E1; }
.hero-card .hero-number { font-size:4.5rem; font-weight:700; letter-spacing:-5px; line-height:1.2; color:#315EFB; }
.hero-card .hero-description { font-size:.9rem; color:#B7C7E1; line-height:1.6; }
.hero-card .hero-rule { border-top:1px solid #385071; margin:22px 0 16px; }
.hero-card .hero-meta { display:flex; justify-content:space-between; font-size:.8rem; color:#D0DDF0; }
[data-testid="stForm"] { background:#FFFFFF; border:1px solid #DFE6F1; border-radius:14px; padding:24px; }
label, [data-testid="stWidgetLabel"] p { color:#40516D !important; font-size:.82rem !important; font-weight:500; }
[data-testid="stNumberInputContainer"], [data-testid="stNumberInputContainer"] > div, [data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="select"] > div { background:#F8FAFE !important; color:#172B4D !important; border-color:#DCE4F0 !important; }
[data-testid="stNumberInputContainer"] input, [data-baseweb="input"] input, [data-baseweb="select"] span, [data-baseweb="select"] div { color:#172B4D !important; -webkit-text-fill-color:#172B4D !important; }
[data-testid="stSelectbox"] [role="combobox"], [data-testid="stSelectbox"] [role="button"] { background:#F8FAFE !important; color:#172B4D !important; border-color:#DCE4F0 !important; }
[data-testid="stSelectbox"] svg { fill:#315EFB !important; color:#315EFB !important; }
[data-testid="stNumberInput"] button { background:#EDF2FC !important; color:#315EFB !important; border:none !important; }
[data-testid="stNumberInput"] button svg { fill:#315EFB !important; color:#315EFB !important; }
[role="option"][aria-selected="true"], [role="option"][data-focused] { background:#EAF0FF !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] { background:#FFFFFF !important; color:#172B4D !important; }
[data-testid="stFormSubmitButton"] button, .stButton button { background:#315EFB !important; color:#FFFFFF !important; border:1px solid #315EFB !important; border-radius:8px; padding:12px 24px; width:100%; font-weight:600; }
[data-testid="stFormSubmitButton"] button:hover { background:#2449C9 !important; }
input:focus-visible, button:focus-visible, a:focus-visible { outline:3px solid #7E9AFF !important; outline-offset:3px; }
[data-testid="stNumberInputContainer"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color:#315EFB !important; box-shadow:0 0 0 2px #315EFB20; }
.stTabs [data-baseweb="tab-list"] { gap:20px; border-bottom:1px solid #DFE6F1; }
.stTabs [data-baseweb="tab"] { color:#5E6E87; background:transparent; }
.stTabs [aria-selected="true"] { color:#315EFB !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#315EFB; }
.feat-badge { display:inline-block; background:#EDF2FF !important; border:1px solid #DCE5FF !important; color:#315EAB !important; border-radius:5px; padding:5px 9px; font-size:.78rem; margin:3px; }
[data-testid="stProgress"] [role="progressbar"] { accent-color:#315EFB; }
.result-card { border-radius:12px !important; }
[id="prediction"], [id="analytics"], [id="model-info"] { scroll-margin-top:24px; }
@media (max-width: 760px) {
 [data-testid="stMainBlockContainer"] { padding:1rem 1rem 3rem; }
 .topnav { flex-wrap:wrap; gap:14px; }
 .nav-links { gap:18px; order:3; width:100%; }
 .nav-status { font-size:.65rem; }
 [data-testid="stForm"] { padding:14px; }
 .result-card { padding:22px !important; }
 h1 { font-size:2.4rem !important; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; } }

/* Dashboard surfaces and responsive presentation; native widgets retain their behavior. */
.topnav { background:#FFFFFF; border:1px solid #DFE6F1; border-radius:16px; padding:18px 24px; box-shadow:0 4px 20px #172b4d05; }
.nav-logo { letter-spacing:-.5px; }
.nav-status { border-radius:100px; }
.section-pill { letter-spacing:2px; font-size:.68rem; }
.hero-card { position:relative; overflow:hidden; padding:34px; border-radius:20px; box-shadow:0 14px 32px #142b5014; }
.hero-card::after { content:""; position:absolute; width:190px; height:190px; border:30px solid #ffffff06; border-radius:50%; right:-90px; top:-75px; pointer-events:none; }
.hero-card .hero-number { color:#FFFFFF; font-size:5rem; letter-spacing:-4px; margin:14px 0; }
.hero-card .hero-description { color:#B7C7E1; }
.hero-card .hero-meta { gap:16px; flex-wrap:wrap; }
[data-testid="stForm"] { padding:28px 32px; border-radius:18px; box-shadow:0 8px 32px #172b4d05; border-top:4px solid #315EFB; }
[data-testid="stForm"] [data-testid="stMarkdownContainer"] > div > span:first-child { font-size:.75rem !important; font-weight:700; color:#315EFB; background:#EDF2FF; border:1px solid #DFE7FF; border-radius:8px; padding:7px 10px; }
[data-testid="stFormSubmitButton"] button { min-height:50px; border-radius:10px; box-shadow:0 4px 12px #315efb20; }
[data-testid="stPlotlyChart"] { background:#FFFFFF; border:1px solid #DFE6F1; border-radius:14px; padding:12px; }
.stat-card { border-top:3px solid #315EFB !important; }
.stTabs [data-baseweb="tab"] { padding:14px 4px; }
@media (max-width:760px) {
 .topnav { padding:16px; }
 .hero-card { padding:24px; }
 .hero-card .hero-number { font-size:3.8rem; }
 [data-testid="stForm"] { padding:16px; }
 .nav-links { flex-wrap:wrap; gap:14px; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<nav class="topnav">
  <div class="nav-logo">
    <svg width="30" height="30" viewBox="0 0 30 30" fill="none">
      <rect width="30" height="30" rx="8" fill="url(#bg-g)"/>
      <circle cx="10" cy="15" r="4" fill="url(#dot-g)" opacity="0.9"/>
      <circle cx="20" cy="10" r="3" fill="url(#dot-g)" opacity="0.7"/>
      <circle cx="20" cy="20" r="3" fill="url(#dot-g)" opacity="0.7"/>
      <line x1="14" y1="13" x2="17" y2="11.5" stroke="rgba(23,43,77,0.5)" stroke-width="1.2"/>
      <line x1="14" y1="17" x2="17" y2="18.5" stroke="rgba(23,43,77,0.5)" stroke-width="1.2"/>
      <defs>
        <linearGradient id="bg-g" x1="0" y1="0" x2="30" y2="30">
          <stop stop-color="#315EFB" stop-opacity="0.25"/>
          <stop offset="1" stop-color="#5373B8" stop-opacity="0.18"/>
        </linearGradient>
        <linearGradient id="dot-g" x1="0" y1="0" x2="30" y2="30">
          <stop stop-color="#315EFB"/><stop offset="1" stop-color="#5373B8"/>
        </linearGradient>
      </defs>
    </svg>
    <span class="logo-text">BagAI</span>
    <span style="color:#687892;font-size:0.82rem;font-weight:400;margin-left:-4px;">&nbsp;· Risk Engine</span>
  </div>
  <ul class="nav-links">
    <li><a href="#" class="active">Home</a></li>
    <li><a href="#prediction">Prediction</a></li>
    <li><a href="#analytics">Analytics</a></li>
    <li><a href="#model-info">Model</a></li>
  </ul>
  <div class="nav-status">
    <div class="status-dot"></div>
    Bagging Engine Live
  </div>
</nav>
""", unsafe_allow_html=True)

# Model error
if not model_loaded:
    st.markdown(f"""
    <div class="glass-card fade-up" style="padding:2.2rem;margin-top:3rem;
         border-color:rgba(183,68,68,0.38);background:rgba(183,68,68,0.06);">
        <h3 style="color:#B74444;margin:0 0 10px 0;">⚠️ Model Load Error</h3>
        <p style="color:#5E6E87;margin:0;font-size:0.95rem;">{model_error}</p>
        <p style="color:#687892;margin:8px 0 0 0;font-size:0.85rem;">
          Make sure <code style="color:#315EFB;">bagging.pkl</code> is in the same directory.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

col_hero, col_visual = st.columns([58, 42], gap="large")

with col_hero:
    st.markdown("""
    <div class="fade-up">
      <div class="section-pill">LOAN RISK WORKSPACE</div>
      <h1 style="font-size:clamp(2.3rem,3.8vw,3.5rem);font-weight:900;line-height:1.11;
                 letter-spacing:-1.8px;margin:0 0 22px 0;color:#172B4D;">
        Every assessment.<br>
        <span class="shimmer-text">A clearer perspective.</span>
      </h1>
      <p style="color:#5E6E87;font-size:1.08rem;line-height:1.72;max-width:510px;margin:0 0 38px 0;">
        Review applicant details, assess default risk, and explore
        prediction insights in one workspace.
      </p>
      <div style="display:flex;gap:16px;flex-wrap:wrap;">
        <a href="#prediction" style="
          display:inline-flex;align-items:center;gap:9px;
          background:#315EFB;
          color:#FFFFFF;text-decoration:none;
          border-radius:8px;padding:12px 24px;
          font-weight:700;font-size:1rem;
          box-shadow:none;
          transition:all 0.3s ease;">New assessment ↗</a>
        <a href="#model-info" style="
          display:inline-flex;align-items:center;gap:9px;
          background:rgba(23,43,77,0.055);
          border:1px solid rgba(23,43,77,0.11);
          color:#172B4D;text-decoration:none;
          border-radius:8px;padding:12px 24px;
          font-weight:600;font-size:1rem;
          transition:all 0.3s ease;">Explore model</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_visual:
    st.markdown("""
    <div class="hero-card">
      <div class="eyebrow">POWERED BY ENSEMBLE LEARNING</div>
      <div class="hero-number">10<span style="font-size:1.2rem;letter-spacing:0;margin-left:12px;">decision trees</span></div>
      <p class="hero-description">Multiple perspectives. One risk assessment.<br>Explore the model's prediction for each applicant.</p>
      <div class="hero-rule"></div>
      <div class="hero-meta"><span>24 encoded features</span><span>Bagging classifier ↗</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px' id='prediction'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="fade-up">
  <div class="section-pill">01 / ASSESSMENT</div>
  <h2 style="font-size:2rem;font-weight:800;letter-spacing:-1.1px;margin:0 0 10px 0;">Applicant assessment</h2>
  <p style="color:#5E6E87;font-size:0.98rem;margin:0 0 32px 0;max-width:560px;">
    Complete the four sections below, then generate a prediction to review the applicant's default risk.
  </p>
</div>
""", unsafe_allow_html=True)

with st.form("bagging_prediction_form", clear_on_submit=False):

    # Section 1: Personal
    st.markdown("""
    <div style="display:flex;align-items:center;gap:11px;padding:18px 14px 10px 14px;
         border-bottom:1px solid rgba(23,43,77,0.055);margin-bottom:20px;">
      <span style="font-size:1.25rem;">01</span>
      <span style="font-weight:700;font-size:1.05rem;color:#172B4D;">Personal Information</span>
    </div>
    """, unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4, gap="medium")
    with p1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35,
                              help="Applicant's current age in years")
    with p2:
        education = st.selectbox("Education Level",
                                 ["High School", "Bachelor's", "Master's", "PhD"],
                                 help="Highest education level attained")
    with p3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with p4:
        has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Section 2: Financial
    st.markdown("""
    <div style="display:flex;align-items:center;gap:11px;padding:18px 14px 10px 14px;
         border-bottom:1px solid rgba(23,43,77,0.055);margin-bottom:20px;">
      <span style="font-size:1.25rem;">02</span>
      <span style="font-weight:700;font-size:1.05rem;color:#172B4D;">Financial Profile</span>
    </div>
    """, unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4, gap="medium")
    with f1:
        income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=1000)
    with f2:
        employment = st.selectbox("Employment Type",
                                  ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    with f3:
        months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=36)
    with f4:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700,
                                       help="FICO range: 300–850")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Section 3: Loan Details
    st.markdown("""
    <div style="display:flex;align-items:center;gap:11px;padding:18px 14px 10px 14px;
         border-bottom:1px solid rgba(23,43,77,0.055);margin-bottom:20px;">
      <span style="font-size:1.25rem;">03</span>
      <span style="font-weight:700;font-size:1.05rem;color:#172B4D;">Loan Application Details</span>
    </div>
    """, unsafe_allow_html=True)
    l1, l2, l3, l4 = st.columns(4, gap="medium")
    with l1:
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000, step=1000)
    with l2:
        loan_term = st.number_input("Term (Months)", min_value=1, max_value=360, value=60)
    with l3:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0,
                                        value=5.5, step=0.1)
    with l4:
        loan_purpose = st.selectbox("Loan Purpose",
                                    ["Business", "Education", "Home", "Other", "Personal"])

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Section 4: Risk Indicators
    st.markdown("""
    <div style="display:flex;align-items:center;gap:11px;padding:18px 14px 10px 14px;
         border-bottom:1px solid rgba(23,43,77,0.055);margin-bottom:20px;">
      <span style="font-size:1.25rem;">04</span>
      <span style="font-weight:700;font-size:1.05rem;color:#172B4D;">Risk Indicators</span>
    </div>
    """, unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4, gap="medium")
    with r1:
        dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0,
                                    value=0.30, step=0.01,
                                    help="Debt-to-Income ratio (0=no debt, 1=full income)")
    with r2:
        num_credit_lines = st.number_input("Active Credit Lines", min_value=0, max_value=50, value=4)
    with r3:
        has_mortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
    with r4:
        has_cosigner = st.selectbox("Has Co-Signer", ["No", "Yes"])

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    _, col_btn, _ = st.columns([2, 5, 2])
    with col_btn:
        submit = st.form_submit_button("Analyze default risk →")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC  (identical preprocessing to run_bag.py — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
if submit:
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    with st.spinner("🎰  Running ensemble analysis across 10 decision trees..."):
        time.sleep(1.5)

        # ── Build 16-column DataFrame matching the Pipeline's ColumnTransformer ──
        # Categorical fields use exact string values as seen in training data.
        # Binary fields (HasMortgage, HasDependents, HasCoSigner) are 0/1 integers.
        input_data = pd.DataFrame([{
            "Age":            int(age),
            "Income":         float(income),
            "LoanAmount":     float(loan_amount),
            "CreditScore":    int(credit_score),
            "MonthsEmployed": int(months_employed),
            "NumCreditLines": int(num_credit_lines),
            "InterestRate":   float(interest_rate),
            "LoanTerm":       int(loan_term),
            "DTIRatio":       float(dti_ratio),
            "Education":      education,          # e.g. "Bachelor's"
            "EmploymentType": employment,          # e.g. "Full-time"
            "MaritalStatus":  marital_status,      # e.g. "Single"
            "HasMortgage":    1 if has_mortgage   == "Yes" else 0,
            "HasDependents":  1 if has_dependents == "Yes" else 0,
            "LoanPurpose":    loan_purpose,        # e.g. "Personal"
            "HasCoSigner":    1 if has_cosigner   == "Yes" else 0,
        }])

        prediction = model.predict(input_data)[0]

        try:
            proba         = model.predict_proba(input_data)[0]
            default_prob  = float(proba[1])
            approval_prob = float(proba[0])
            has_proba     = True
        except Exception:
            has_proba     = False
            default_prob  = None
            approval_prob = None

        # Store in session history
        st.session_state.total_predictions += 1
        st.session_state.prediction_history.append({
            "prediction":   int(prediction),
            "credit_score": credit_score,
            "income":       income,
            "loan_amount":  loan_amount,
            "dti_ratio":    dti_ratio,
            "prob_default": default_prob,
        })

    # Build result variables
    if prediction == 1:
        confidence      = default_prob  if has_proba else None
        result_color    = "#B74444"
        result_bg       = "rgba(183,68,68,0.08)"
        result_border   = "rgba(183,68,68,0.35)"
        result_icon     = "⚠️"
        result_title    = "High Risk — Manual Review Required"
        result_badge    = "🔴 &nbsp;DECLINED"
        result_badge_bg = "rgba(183,68,68,0.14)"
        result_badge_c  = "#B74444"
        result_text     = ("The Bagging Ensemble has identified elevated default risk. "
                           "The majority of the 10 decision trees voted for a default outcome. "
                           "The applicant's financial profile does not meet automated approval thresholds — "
                           "escalation to a human underwriter is strongly recommended.")
    else:
        confidence      = approval_prob if has_proba else None
        result_color    = "#24754B"
        result_bg       = "rgba(36,117,75,0.08)"
        result_border   = "rgba(36,117,75,0.35)"
        result_icon     = "✅"
        result_title    = "Low Risk — Approved for Processing"
        result_badge    = "🟢 &nbsp;APPROVED"
        result_badge_bg = "rgba(36,117,75,0.14)"
        result_badge_c  = "#24754B"
        result_text     = ("Excellent! The Bagging Ensemble has given a majority-vote approval. "
                           "The applicant's profile demonstrates a strong repayment profile across all 10 decision trees. "
                           "All key financial indicators point to a low probability of default — "
                           "recommended for immediate automated loan processing.")
        st.balloons()

    conf_pct = f"{confidence * 100:.1f}%" if confidence is not None else "N/A"
    conf_int = int(confidence * 100)       if confidence is not None else 0

    # Premium result card
    st.markdown(f"""
    <div class="result-card" style="
        background:{result_bg};border:1.5px solid {result_border};
        border-radius:26px;padding:38px 42px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:14px;margin-bottom:26px;">
        <div style="display:flex;align-items:center;gap:16px;">
          <div style="font-size:2.5rem;line-height:1;">{result_icon}</div>
          <div>
            <div style="font-size:1.22rem;font-weight:800;color:#172B4D;letter-spacing:-0.5px;">
              {result_title}</div>
            <div style="font-size:0.84rem;color:#5E6E87;margin-top:3px;">
              Bagging Classifier · 10 Decision Trees · 24 Features</div>
          </div>
        </div>
        <div style="background:{result_badge_bg};color:{result_badge_c};
             border:1px solid {result_border};border-radius:100px;padding:9px 22px;
             font-weight:700;font-size:0.86rem;letter-spacing:1.2px;">{result_badge}</div>
      </div>
      <p style="color:#40516D;font-size:0.97rem;line-height:1.72;margin:0 0 28px 0;">{result_text}</p>
      <div>
        <div style="display:flex;justify-content:space-between;margin-bottom:9px;">
          <span style="font-size:0.84rem;color:#5E6E87;font-weight:500;">Ensemble Confidence Score</span>
          <span style="font-size:0.96rem;font-weight:800;color:{result_color};">{conf_pct}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if has_proba:
        st.progress(conf_int / 100)

    # Key metric mini-cards
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    for col, label, val, color in [
        (m1, "Credit Score",  f"{credit_score}",   "#5373B8"),
        (m2, "Annual Income", f"${income:,}",       "#315EFB"),
        (m3, "Loan Amount",   f"${loan_amount:,}",  "#A36A20"),
        (m4, "DTI Ratio",     f"{dti_ratio:.0%}",   result_color),
    ]:
        with col:
            st.markdown(f"""
            <div class="glass-card" style="padding:20px;text-align:center;">
              <div style="font-size:0.73rem;color:#5E6E87;text-transform:uppercase;
                   letter-spacing:1.1px;margin-bottom:7px;">{label}</div>
              <div style="font-size:1.38rem;font-weight:800;color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # Plotly charts
    if has_proba:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2, gap="large")

        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_prob * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#172B4D", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                              "tickcolor": "rgba(23,43,77,0.2)",
                              "tickfont": {"color": "rgba(23,43,77,0.4)", "size": 11}},
                    "bar": {"color": "#B74444" if prediction == 1 else "#24754B", "thickness": 0.28},
                    "bgcolor": "rgba(23,43,77,0.03)", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35],   "color": "rgba(36,117,75,0.12)"},
                        {"range": [35, 65],  "color": "rgba(163,106,32,0.10)"},
                        {"range": [65, 100], "color": "rgba(183,68,68,0.12)"},
                    ],
                    "threshold": {"line": {"color": "#A36A20", "width": 2},
                                  "thickness": 0.75, "value": 50},
                },
                title={"text": "Default Probability", "font": {"size": 14, "color": "#5E6E87", "family": "Inter"}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"}, height=250,
                margin=dict(t=60, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, width="stretch")

        with g2:
            fig_donut = go.Figure(go.Pie(
                labels=["Repayment Probability", "Default Probability"],
                values=[approval_prob * 100, default_prob * 100],
                hole=0.58,
                marker=dict(colors=["#24754B", "#B74444"],
                            line=dict(color="#F3F6FB", width=2)),
                textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#172B4D"},
                height=250, margin=dict(t=30, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(font=dict(color="#5E6E87", size=12), bgcolor="rgba(0,0,0,0)",
                            orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
                annotations=[dict(text="HIGH RISK" if prediction == 1 else "LOW RISK",
                                  font=dict(size=13, color=result_color, family="Inter"),
                                  showarrow=False)],
            )
            st.plotly_chart(fig_donut, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:36px' id='analytics'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="fade-up">
  <div class="section-pill">📈 &nbsp;Analytics</div>
  <h2 style="font-size:2rem;font-weight:800;letter-spacing:-1.1px;margin:0 0 10px 0;">Session Analytics</h2>
  <p style="color:#5E6E87;font-size:0.97rem;margin:0 0 32px 0;">
    Visual insights from predictions made in this session.
  </p>
</div>
""", unsafe_allow_html=True)

if len(st.session_state.prediction_history) == 0:
    st.markdown("""
    <div style="background:rgba(23,43,77,0.025);border:1px solid rgba(23,43,77,0.06);
         border-radius:18px;padding:48px;text-align:center;">
      <div style="font-size:2.8rem;margin-bottom:14px;">📊</div>
      <div style="font-size:1.05rem;font-weight:600;color:#172B4D;margin-bottom:8px;">No predictions yet</div>
      <div style="font-size:0.9rem;color:#687892;">
        Run your first prediction above — analytics will appear here automatically.
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    hist       = st.session_state.prediction_history
    n_total    = len(hist)
    n_default  = sum(1 for h in hist if h["prediction"] == 1)
    n_approved = n_total - n_default

    an1, an2, an3, an4 = st.columns(4, gap="medium")
    _ac = ("background:rgba(23,43,77,0.03);border:1px solid rgba(23,43,77,0.07);"
           "border-radius:18px;padding:24px;text-align:center;")
    for col, label, val, color in [
        (an1, "Total Predictions", str(n_total),     "#315EFB"),
        (an2, "Approved",          str(n_approved),  "#24754B"),
        (an3, "Declined",          str(n_default),   "#B74444"),
        (an4, "Approval Rate",
         f"{(n_approved/n_total*100):.1f}%",         "#5373B8"),
    ]:
        with col:
            st.markdown(f"""
            <div style="{_ac}">
              <div style="font-size:0.75rem;color:#5E6E87;text-transform:uppercase;
                   letter-spacing:1.1px;margin-bottom:8px;">{label}</div>
              <div style="font-size:1.8rem;font-weight:800;color:{color};">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    al, ar = st.columns(2, gap="large")

    with al:
        fig_bar = go.Figure(go.Bar(
            x=["Approved", "Declined"],
            y=[n_approved, n_default],
            marker=dict(color=["#24754B", "#B74444"],
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[str(n_approved), str(n_default)],
            textposition="outside",
            textfont=dict(color="#172B4D", size=13, family="Inter"),
            width=0.45,
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#5E6E87"),
            title=dict(text="Prediction Distribution", font=dict(size=15, color="#172B4D")),
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#5E6E87", size=12)),
            yaxis=dict(showgrid=True, gridcolor="rgba(23,43,77,0.05)",
                       zeroline=False, tickfont=dict(color="#5E6E87", size=11)),
            height=300, margin=dict(t=50, b=30, l=30, r=20),
        )
        st.plotly_chart(fig_bar, width="stretch")

    with ar:
        proba_vals = [h["prob_default"] for h in hist if h["prob_default"] is not None]
        cs_vals    = [h["credit_score"] for h in hist if h["prob_default"] is not None]

        if proba_vals:
            colors_sc = ["#B74444" if p > 0.5 else "#24754B" for p in proba_vals]
            fig_sc = go.Figure(go.Scatter(
                x=cs_vals, y=[p * 100 for p in proba_vals],
                mode="markers",
                marker=dict(size=12, color=colors_sc, opacity=0.85,
                            line=dict(color="rgba(23,43,77,0.2)", width=1)),
                hovertemplate="Credit Score: %{x}<br>Default Prob: %{y:.1f}%<extra></extra>",
            ))
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#5E6E87"),
                title=dict(text="Credit Score vs Default Probability",
                           font=dict(size=15, color="#172B4D")),
                xaxis=dict(title="Credit Score", showgrid=True,
                           gridcolor="rgba(23,43,77,0.05)",
                           zeroline=False, tickfont=dict(color="#5E6E87", size=11)),
                yaxis=dict(title="Default Prob (%)", showgrid=True,
                           gridcolor="rgba(23,43,77,0.05)",
                           zeroline=False, tickfont=dict(color="#5E6E87", size=11)),
                height=300, margin=dict(t=50, b=30, l=50, r=20),
            )
            st.plotly_chart(fig_sc, width="stretch")
        else:
            st.markdown("""
            <div style="padding:60px;text-align:center;color:#687892;">
              Probability data will appear after predictions.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL OVERVIEW STAT CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="fade-up delay-1">
  <div class="section-pill">📊 &nbsp;Model Overview</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="medium")
_card = ("background:rgba(23,43,77,0.04);backdrop-filter:blur(20px);"
         "-webkit-backdrop-filter:blur(20px);border:1px solid rgba(23,43,77,0.07);"
         "border-radius:20px;padding:26px 20px;text-align:center;cursor:default;")
for col, icon, stat, label, color in [
    (c1, "🎰", "Bagging",  "Algorithm",       "#315EFB"),
    (c2, "🌲", "10",       "Base Trees",       "#5373B8"),
    (c3, "📐", "24",       "Input Features",   "#24754B"),
    (c4, "⚡", "< 2s",     "Response Time",    "#A36A20"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-card fade-up" style="{_card}">
          <div style="font-size:2.1rem;margin-bottom:12px;">{icon}</div>
          <div style="font-size:1.55rem;font-weight:800;color:{color};letter-spacing:-0.4px;margin-bottom:5px;">{stat}</div>
          <div style="font-size:0.79rem;color:#5E6E87;font-weight:500;text-transform:uppercase;letter-spacing:0.9px;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="fade-up">
  <div class="section-pill">🔄 &nbsp;Workflow</div>
  <h2 style="font-size:2rem;font-weight:800;letter-spacing:-1.1px;margin:0 0 38px 0;">How It Works</h2>
</div>
""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="large")
for col, num, icon, title, desc, color in [
    (s1,"01","📋","Enter Applicant Data",
     "Fill in the applicant's personal, financial, and loan details in the structured smart form.","#315EFB"),
    (s2,"02","🎰","Bagging Ensemble Analysis",
     "10 Decision Trees independently analyze 24 features and vote on the risk outcome.","#5373B8"),
    (s3,"03","📊","Majority-Vote Result",
     "The ensemble majority vote produces a robust, unbiased prediction with confidence scoring.","#24754B"),
]:
    r,g,b = tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    with col:
        st.markdown(f"""
        <div class="glass-card fade-up" style="padding:30px;text-align:center;">
          <div style="width:54px;height:54px;border-radius:15px;
            background:rgba({r},{g},{b},0.14);border:1px solid rgba({r},{g},{b},0.32);
            display:flex;align-items:center;justify-content:center;
            font-size:1.5rem;margin:0 auto 18px auto;">{icon}</div>
          <div style="font-size:0.70rem;color:{color};font-weight:700;
               letter-spacing:2.2px;text-transform:uppercase;margin-bottom:10px;">Step {num}</div>
          <div style="font-size:1.08rem;font-weight:700;color:#172B4D;margin-bottom:12px;">{title}</div>
          <div style="font-size:0.88rem;color:#5E6E87;line-height:1.65;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFORMATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:36px' id='model-info'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="fade-up">
  <div class="section-pill">🤖 &nbsp;Model Information</div>
  <h2 style="font-size:2rem;font-weight:800;letter-spacing:-1.1px;margin:0 0 28px 0;">About the Bagging Model</h2>
</div>
""", unsafe_allow_html=True)

tab_spec, tab_feats, tab_data = st.tabs(["⚙️ Specifications", "📐 Features", "📋 Dataset"])

with tab_spec:
    ts1, ts2 = st.columns(2, gap="large")
    with ts1:
        st.markdown("""
        <div class="glass-card" style="padding:30px;height:100%;">
          <h4 style="color:#172B4D;font-size:1.05rem;font-weight:700;margin:0 0 22px 0;">⚙️ Technical Specifications</h4>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Algorithm</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">BaggingClassifier</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Base Estimator</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">DecisionTreeClassifier</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">n_estimators</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">10</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Sampling</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">Bootstrap (with replacement)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Class Balancing</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">SMOTE (training only)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Prediction Type</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">Binary Classification</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Target Variable</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">Default (0 = No, 1 = Yes)</td>
            </tr>
            <tr>
              <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Framework</td>
              <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">scikit-learn</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with ts2:
        st.markdown("""
        <div class="glass-card" style="padding:30px;height:100%;">
          <h4 style="color:#172B4D;font-size:1.05rem;font-weight:700;margin:0 0 22px 0;">🎰 How Bagging Works</h4>
          <div style="display:flex;flex-direction:column;gap:14px;">
            <div style="padding:14px 18px;background:rgba(49,94,251,0.10);
                 border:1px solid rgba(49,94,251,0.24);border-radius:12px;">
              <div style="font-size:0.74rem;color:#315EFB;font-weight:700;
                   text-transform:uppercase;letter-spacing:1.1px;margin-bottom:5px;">1 — Bootstrap Sampling</div>
              <div style="font-size:0.87rem;color:#40516D;line-height:1.55;">
                Each of the 10 trees trains on a random bootstrap sample drawn with replacement
                from the SMOTE-balanced training dataset.
              </div>
            </div>
            <div style="padding:14px 18px;background:rgba(83,115,184,0.08);
                 border:1px solid rgba(83,115,184,0.20);border-radius:12px;">
              <div style="font-size:0.74rem;color:#5373B8;font-weight:700;
                   text-transform:uppercase;letter-spacing:1.1px;margin-bottom:5px;">2 — Parallel Tree Training</div>
              <div style="font-size:0.87rem;color:#40516D;line-height:1.55;">
                All 10 Decision Trees are trained independently and in parallel,
                each producing a diverse set of decision rules.
              </div>
            </div>
            <div style="padding:14px 18px;background:rgba(36,117,75,0.07);
                 border:1px solid rgba(36,117,75,0.20);border-radius:12px;">
              <div style="font-size:0.74rem;color:#24754B;font-weight:700;
                   text-transform:uppercase;letter-spacing:1.1px;margin-bottom:5px;">3 — Majority Vote</div>
              <div style="font-size:0.87rem;color:#40516D;line-height:1.55;">
                At inference time, all 10 trees vote. The class receiving the most votes
                wins — reducing variance and improving robustness.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with tab_feats:
    st.markdown("""
    <div class="glass-card" style="padding:30px;">
      <h4 style="color:#172B4D;font-size:1.05rem;font-weight:700;margin:0 0 20px 0;">📐 Feature Engineering — 24 Total Features</h4>
      <div style="display:flex;flex-direction:column;gap:16px;">
        <div style="padding:16px 20px;background:rgba(49,94,251,0.09);
             border:1px solid rgba(49,94,251,0.22);border-radius:14px;">
          <div style="font-size:0.76rem;color:#315EFB;font-weight:700;
               text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">Numeric Features (9)</div>
          <div>
            <span class="feat-badge">Age</span><span class="feat-badge">Income</span>
            <span class="feat-badge">LoanAmount</span><span class="feat-badge">CreditScore</span>
            <span class="feat-badge">MonthsEmployed</span><span class="feat-badge">NumCreditLines</span>
            <span class="feat-badge">InterestRate</span><span class="feat-badge">LoanTerm</span>
            <span class="feat-badge">DTIRatio</span>
          </div>
        </div>
        <div style="padding:16px 20px;background:rgba(83,115,184,0.07);
             border:1px solid rgba(83,115,184,0.18);border-radius:14px;">
          <div style="font-size:0.76rem;color:#5373B8;font-weight:700;
               text-transform:uppercase;letter-spacing:1.3px;margin-bottom:10px;">One-Hot Encoded (15)</div>
          <div>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">Education_HighSchool</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">Education_Master's</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">Education_PhD</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">Employment_PartTime</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">Employment_SelfEmployed</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">Employment_Unemployed</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">MaritalStatus_Married</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">MaritalStatus_Single</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">HasMortgage</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">HasDependents</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">LoanPurpose_Business</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">LoanPurpose_Education</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">LoanPurpose_Home</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">LoanPurpose_Other</span>
            <span class="feat-badge" style="background:rgba(83,115,184,0.09);border-color:rgba(83,115,184,0.20);color:#5373B8;">HasCoSigner</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with tab_data:
    st.markdown("""
    <div class="glass-card" style="padding:30px;">
      <h4 style="color:#172B4D;font-size:1.05rem;font-weight:700;margin:0 0 20px 0;">📋 Dataset Information</h4>
      <table style="width:100%;border-collapse:collapse;">
        <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
          <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Dataset File</td>
          <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">Loan_data.csv</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
          <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Train / Test Split</td>
          <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">80% / 20% (random_state=42)</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
          <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Class Balancing</td>
          <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">SMOTE (random_state=42)</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
          <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Encoding Strategy</td>
          <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">One-Hot (drop_first=True)</td>
        </tr>
        <tr style="border-bottom:1px solid rgba(23,43,77,0.055);">
          <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">Feature Scaling</td>
          <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">StandardScaler on numeric columns</td>
        </tr>
        <tr>
          <td style="color:#5E6E87;padding:11px 0;font-size:0.89rem;">LoanID Columns</td>
          <td style="color:#5373B8;font-weight:600;font-size:0.89rem;text-align:right;">Dropped before training</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:30px;border-top:1px solid rgba(23,43,77,0.055);">
  <div style="font-size:1.1rem;font-weight:800;letter-spacing:-0.5px;
    background:linear-gradient(135deg,#315EFB,#5373B8);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin-bottom:9px;">
    BagAI · Loan Default Risk Intelligence
  </div>
  <div style="color:#687892;font-size:0.83rem;">
    Built with &nbsp;🐍 Python &nbsp;·&nbsp; 🎈 Streamlit &nbsp;·&nbsp;
    🎰 scikit-learn Bagging &nbsp;·&nbsp; 📊 Plotly &nbsp;·&nbsp; Machine Learning
  </div>
</div>
""", unsafe_allow_html=True)
