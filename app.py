"""
Streamlit UI — Compound Toxicity Prediction System
Run with:  streamlit run app.py

Tabs: Dataset -> EDA -> Model Results -> Predict
Theme: Dark Glassmorphism + Mint Accent
"""

import os
import json
import pickle
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw


st.set_page_config(page_title="Drug Discovery Prediction", page_icon="💊", layout="wide")

# ---------------------------------------------------------------------
# Style — Dark Glassmorphism + Mint Accent (Syne / DM Sans)
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg-base:        #0d0f14;
  --bg-card:        rgba(255,255,255,0.04);
  --bg-input:       rgba(255,255,255,0.06);
  --border:         rgba(255,255,255,0.08);
  --border-hover:   rgba(255,255,255,0.18);

  --accent:         #3bffa0;
  --accent-dim:     rgba(59,255,160,0.12);
  --accent-glow:    0 0 24px rgba(59,255,160,0.25);

  --text-primary:   #f0f4f0;
  --text-secondary: #7e8d85;
  --text-muted:     #4a5550;

  --danger:         #ff5c5c;
  --danger-dim:     rgba(255,92,92,0.12);
  --warning:        #ffc145;

  --radius-sm:  8px;
  --radius-md:  14px;
  --radius-lg:  22px;

  --transition: 0.22s cubic-bezier(0.4, 0, 0.2, 1);
  --font-head:  'Syne', sans-serif;
  --font-body:  'DM Sans', sans-serif;
}

html, body, [class*="css"], [class*="st-"], .stApp, .stApp * ,
button, input, textarea, select, .stMarkdown, .stMarkdown p,
[data-testid="stSidebar"] * , [data-testid="stWidgetLabel"] * {
  font-family: var(--font-body) !important;
}

[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
}

:root, .stApp {
  --font: 'DM Sans', sans-serif;
  --font-sans: 'DM Sans', sans-serif;
}

h1, h1 *, h2, h2 *, h3, h3 *,
.info-card h3, .acc-value, .nav-brand {
  font-family: var(--font-head) !important;
}

.stApp {
  background-color: var(--bg-base);
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(59,255,160,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 60% 40% at 80% 90%, rgba(59,160,255,0.05) 0%, transparent 60%);
  color: var(--text-primary);
}

/* Headings */
h1 {
  font-family: var(--font-head) !important;
  color: var(--text-primary) !important;
  font-weight: 800 !important;
  letter-spacing: -0.5px;
  background: linear-gradient(120deg, #f0f4f0 30%, var(--accent) 70%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
h2, h3 {
  font-family: var(--font-head) !important;
  color: var(--text-primary) !important;
  font-weight: 700 !important;
}
h4 { color: var(--text-secondary) !important; font-weight: 400 !important; }
p, span, div, label { color: var(--text-primary); }
hr { border-color: var(--border) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background-color: #0a0c10;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Info card */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 30px;
    margin: 10px 0 22px 0;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
.info-card h3 { color: var(--text-primary) !important; }
.info-card p, .info-card li { color: var(--text-secondary) !important; }
.check-item { color: var(--accent) !important; font-weight: 500; margin: 6px 0; }

/* Accuracy cards */
.acc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-bottom: 16px;
    text-align: center;
    transition: var(--transition);
}
.acc-card:hover { border-color: var(--border-hover); background: rgba(255,255,255,0.06); }
.acc-label { color: var(--text-secondary) !important; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.acc-value { color: var(--accent) !important; font-family: var(--font-head); font-size: 2.1rem; font-weight: 800; }

/* Sidebar badge + model rows */
.sidebar-badge {
    background: var(--accent-dim); color: var(--accent) !important; border-radius: var(--radius-sm);
    padding: 6px 12px; font-size: 0.85rem; font-weight: 600;
    display: inline-block; margin-bottom: 18px;
    border: 1px solid rgba(59,255,160,0.25);
}
.model-row { padding: 6px 0; color: var(--text-primary) !important; font-weight: 500; }

/* Verdict pills */
.verdict-toxic {
  display: inline-block;
  background: var(--danger-dim); color: var(--danger) !important;
  padding: 10px 16px; border-radius: var(--radius-sm); font-weight: 700;
  line-height: 1.4;
  border: 1px solid rgba(255,92,92,0.25);
}
.verdict-safe {
  display: inline-block;
  background: var(--accent-dim); color: var(--accent) !important;
  padding: 10px 16px; border-radius: var(--radius-sm); font-weight: 700;
  line-height: 1.4;
  border: 1px solid rgba(59,255,160,0.25);
}

/* Inputs, selects, radio, buttons */
[data-testid="stTextInputRootElement"],
.stTextInput input,
.stSelectbox > div, .stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea {
  background-color: var(--bg-input) !important;
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-size: 15.5px !important;
  box-shadow: none !important;
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextInputRootElement"]:hover,
.stSelectbox div[role="group"][data-rac]:focus-within,
.stSelectbox div[role="group"][data-rac]:hover {
  box-shadow: var(--accent-glow) !important;
  border-color: var(--accent) !important;
  transition: box-shadow 0.15s ease, border-color 0.15s ease;
}

/* Override native red validation border on the selectbox */
.stSelectbox div[role="group"][data-rac],
.stSelectbox div[role="group"][data-rac][aria-invalid="true"] {
  border-color: var(--border) !important;
}
.stSelectbox div[role="group"][data-rac]:focus-within,
.stSelectbox div[role="group"][data-rac]:focus-within[aria-invalid="true"] {
  border-color: var(--accent) !important;
}

/* Compound name / model selectors — wider + bigger text */
.stSelectbox {
  max-width: none;
  width: 100%;
}
.stSelectbox div[data-baseweb="select"] > div {
  min-height: 46px !important;
}
.stSelectbox div[data-baseweb="select"] * {
  font-size: 16px !important;
}

/* Run Prediction button — full row width + bigger semibold text */
.stButton, .stFormSubmitButton { max-width: none; width: 100%; }
.stButton > button, .stFormSubmitButton > button {
  background: var(--accent) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 900 !important;
  font-size: 24px !important;
  padding: 16px 18px !important;
  transition: var(--transition);
}
.stButton > button, .stButton > button *,
.stFormSubmitButton > button, .stFormSubmitButton > button * {
  color: #0d1a12 !important;
  fill: #0d1a12 !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  background: #5fffb4 !important;
  box-shadow: var(--accent-glow) !important;
  transform: translateY(-2px);
}
div[role="radiogroup"] label { color: var(--text-primary) !important; }
[data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

/* Accuracy percentage numbers — force heading font + slightly bigger */
.stApp .acc-value {
  font-family: var(--font-head) !important;
  font-size: 2.3rem !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px 20px;
  color: var(--text-primary) !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent-dim) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
 
.react-aria-SelectionIndicator {
  background: var(--accent) !important;
  background-color: var(--accent) !important;
}

/* Mobile Optimization */
@media (max-width: 768px) {
  .info-card {
    padding: 16px 18px;
    margin: 8px 0 16px 0;
  }
  .info-card p, .info-card li {
    line-height: 1.65 !important;
    font-size: 0.95rem;
  }
  .info-card h3 {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    line-height: 1.3;
  }
  h1 {
    font-size: 1.85rem !important;
    line-height: 1.25 !important;
  }
  h4 {
    font-size: 0.95rem !important;
    padding: 0 6px;
  }
  .acc-card {
    margin-bottom: 12px;
  }
  .acc-value {
    font-size: 2rem !important;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 8px 12px;
    font-size: 0.85rem !important;
  }
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
  overflow: hidden !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def show_img(path: str):
    st.image(read_file_bytes(path))


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    return pd.read_csv("Dataset/tox21.csv")


@st.cache_data(show_spinner=False)
def load_custom_compounds_cached(mtime: float) -> dict:
    return load_custom_compounds()


# ---------------------------------------------------------------------
# Load trained models
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("models/models.pkl", "rb") as f:
        art = pickle.load(f)
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return art, metrics

art, metrics = load_artifacts()
models = art["models"]
scaler = art["scaler"]
feature_names = art["feature_names"]

MODEL_ICONS = {"Random Forest": "🌲", "SVM (RBF kernel)": "🧬", "Logistic Regression": "📈"}

# ---------------------------------------------------------------------
# Named compound library (English names -> SMILES, hidden from the user)
# ---------------------------------------------------------------------
DRUG_LIBRARY = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Paracetamol (Acetaminophen)": "CC(=O)NC1=CC=C(C=C1)O",
    "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "Amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "Omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "Diphenhydramine": "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "Loratadine": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
    "Warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "Diazepam": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "Morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Glucose": "OCC1OC(O)C(O)C(O)C1O",
    "Vitamin C (Ascorbic acid)": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
    "Naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "Sildenafil (Viagra)": "CCCC1=NN(C)C2=C1NC(=NC2=O)C1=CC(=CC=C1OCC)S(=O)(=O)N1CCN(C)CC1",
    "Fluoxetine (Prozac)": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "Sertraline (Zoloft)": "CNC1CCC(c2ccc(Cl)c(Cl)c2)c2ccccc21",
    "Lisinopril": "CCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)N1CCC[C@H]1C(=O)O)C(=O)O",
    "Simvastatin": "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
    "Losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "Albuterol (Salbutamol)": "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",
    "Methotrexate": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "Chloroquine": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
    "Thalidomide": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
}

# ---------------------------------------------------------------------
# Custom compounds saved by the user (persisted as a structured CSV table)
# Columns: Compound Name, SMILES
# ---------------------------------------------------------------------
CUSTOM_COMPOUNDS_FILE = "Dataset/custom_compounds.csv"


def load_custom_compounds() -> dict:
    """Read saved custom compounds from the CSV table, if it exists."""
    if not os.path.exists(CUSTOM_COMPOUNDS_FILE):
        return {}
    df_custom = pd.read_csv(CUSTOM_COMPOUNDS_FILE)
    df_custom = df_custom.dropna(subset=["Compound Name", "SMILES"])
    return dict(zip(df_custom["Compound Name"].str.strip(), df_custom["SMILES"].str.strip()))


def save_custom_compound(name: str, smiles: str):
    """Append a new compound as a row in the CSV table, skipping duplicate names."""
    os.makedirs(os.path.dirname(CUSTOM_COMPOUNDS_FILE), exist_ok=True)
    existing = load_custom_compounds()
    if name in existing or name in DRUG_LIBRARY:
        return False  # name already used, don't overwrite

    new_row = pd.DataFrame([{"Compound Name": name, "SMILES": smiles}])
    if os.path.exists(CUSTOM_COMPOUNDS_FILE):
        new_row.to_csv(CUSTOM_COMPOUNDS_FILE, mode="a", header=False, index=False)
    else:
        new_row.to_csv(CUSTOM_COMPOUNDS_FILE, mode="w", header=True, index=False)
    return True


def get_full_library() -> dict:
    """Built-in drug library + any user-saved custom compounds."""
    mtime = os.path.getmtime(CUSTOM_COMPOUNDS_FILE) if os.path.exists(CUSTOM_COMPOUNDS_FILE) else 0.0
    return {**DRUG_LIBRARY, **load_custom_compounds_cached(mtime)}



def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    feats = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "RingCount": Descriptors.RingCount(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "MolMR": Descriptors.MolMR(mol),
    }
    return feats, mol
def check_drug_likeness(feats):
    """
    Lipinski's Rule of Five — the gold standard for drug-likeness.
    A drug candidate should have:
      - Molecular Weight <= 500 Da
      - LogP <= 5
      - H-bond Donors <= 5
      - H-bond Acceptors <= 10
    Compounds with 0-1 violations are considered drug-like.
    """
    rules = {
        "Molecular Weight ≤ 500 Da":      (feats["MolWt"] <= 500,          f"{feats['MolWt']:.1f} Da"),
        "LogP ≤ 5 (lipophilicity)":       (feats["LogP"] <= 5,             f"{feats['LogP']:.2f}"),
        "H-bond Donors ≤ 5":              (feats["NumHDonors"] <= 5,       f"{int(feats['NumHDonors'])}"),
        "H-bond Acceptors ≤ 10":          (feats["NumHAcceptors"] <= 10,   f"{int(feats['NumHAcceptors'])}"),
    }
    violations = sum(1 for passed, _ in rules.values() if not passed)
    return rules, violations


# =======================================================================
# SIDEBAR
# =======================================================================
with st.sidebar:
    st.markdown("<div style='font-size:48px; text-align:center;'>💊</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Drug Discovery</h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='sidebar-badge'>Data Mining Project</span></div>",
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ ML Models**")
    for name in models.keys():
        st.markdown(f"<div class='model-row'>{MODEL_ICONS.get(name,'🔹')} {name}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🔬 Screening Criteria**")
    st.markdown("<div class='model-row'>⚠️ Mitochondrial Toxicity</div>", unsafe_allow_html=True)
    st.markdown("<div class='model-row'>💊 Lipinski's Rule of 5</div>", unsafe_allow_html=True)
    st.markdown("<div class='model-row'>🧬 Molecular Descriptors</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🔎 Get Compounds Names & SMILES**")
    st.markdown(
        "<div class='model-row' style='display:flex; align-items:center; gap:8px;'>"
        "<img src='https://www.google.com/s2/favicons?sz=32&domain=litfl.com' "
        "style='width:18px; height:18px; border-radius:4px;'> "
        "<a href='https://litfl.com/bscc/top-200-drugs-pharmacology/' target='_blank' "
        "style='color:var(--accent) !important; text-decoration:none;'>Top 200 Drugs List</a></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='model-row' style='display:flex; align-items:center; gap:8px;'>"
        "<img src='https://www.google.com/s2/favicons?sz=32&domain=pubchem.ncbi.nlm.nih.gov' "
        "style='width:18px; height:18px; border-radius:4px;'> "
        "<a href='https://pubchem.ncbi.nlm.nih.gov/' target='_blank' "
        "style='color:var(--accent) !important; text-decoration:none;'>PubChem — Find SMILES</a></div>",
        unsafe_allow_html=True
    )


# =======================================================================
# MAIN — title + project info card
# =======================================================================
st.markdown(
    "<h1 style='text-align:center;'>"
    "<span style='-webkit-text-fill-color: initial; background: none;'>💊</span> "
    "Drug Discovery Prediction"
    "</h1>",
    unsafe_allow_html=True
)
st.markdown("<h4 style='text-align:center; font-weight:400;'>"
            "Predicting Drug Candidate Viability with Machine Learning</h4>", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-card">
  <h3>📚 Project Information</h3>
  <p>This project applies a full data mining pipeline — data cleaning, feature extraction, exploratory data analysis, 
  and classification — to a real-world dataset (NIH/EPA Tox21). SMILES structures are converted into 12 molecular 
  descriptors using RDKit, then three classification models (Random Forest, SVM, Logistic Regression) are trained 
  to predict mitochondrial toxicity. The compound's drug-likeness is also assessed using Lipinski's Rule of Five. 
  The result: a data-driven screening report for any compound.</p>
  <h3 style="margin-top:18px;">⚙️ Machine Learning Models</h3>
  {''.join(f"<div class='check-item'>✓ {name}</div>" for name in models.keys())}
</div>
""", unsafe_allow_html=True)

# Accuracy cards
cols = st.columns(len(models))
for col, (name, m) in zip(cols, metrics.items()):
    with col:
        st.markdown(f"""
        <div class="acc-card">
            <div class="acc-label">{name}</div>
            <div class="acc-value">{m['accuracy']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# =======================================================================
# TABS — Dataset / EDA / Results / Predict
# =======================================================================
tab_dataset, tab_eda, tab_fpm, tab_results, tab_predict = st.tabs(
    ["🗃️ Dataset", "📊 EDA", "🧩 Frequent Pattern Mining", "📈 Model Results", "🤖 Predict"]
)

# ---------- TAB 1: Dataset ----------
with tab_dataset:
    st.markdown("### 🗃️ Dataset Overview")
    st.markdown("""
    **Source:** Tox21 (NIH/EPA) — real-world chemical toxicity dataset from Kaggle.
    - **Target:** `SR-MMP` (Stress Response — Mitochondrial Membrane Potential disruption)
    - **Input:** SMILES strings → 12 molecular descriptors extracted with RDKit
    - **Task:** Binary classification (Toxic = 1, Non-toxic = 0)
    """)
    try:
        raw = load_raw_dataset()
        st.markdown(f"**Total rows:** {len(raw):,} &nbsp;&nbsp;&nbsp; **Columns:** {raw.shape[1]}")
        st.markdown("#### First 10 rows")
        st.dataframe(raw.head(10), use_container_width=True)
        st.markdown("#### Missing values per column")
        st.dataframe(raw.isnull().sum().rename("missing").to_frame(), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load tox21.csv: {e}")

# ---------- TAB 2: EDA ----------
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis")

    st.markdown("#### Histograms")
    try:
        show_img("outputs/eda/histograms.png")
    except:
        st.info("Run `python train_models.py` first to generate this chart.")

    st.markdown("#### Bar Chart")
    try:
        show_img("outputs/eda/bar_chart.png")
    except:
        st.info("Run `python train_models.py` first.")

    st.markdown("#### Scatter Plot")
    try:
        show_img("outputs/eda/scatter_plot.png")
    except:
        st.info("Run `python train_models.py` first.")

    st.markdown("#### Box Plot")
    try:
        show_img("outputs/eda/box_plot.png")
    except:
        st.info("Run `python train_models.py` first.")

    st.markdown("#### Correlation Matrix & Heatmap")
    try:
        show_img("outputs/eda/correlation_heatmap.png")
    except:
        st.info("Run `python train_models.py` first.")

    st.markdown("#### Identifying Outliers")
    try:
        show_img("outputs/eda/outliers.png")

    except:
        st.info("Run `python train_models.py` first.")

    st.markdown("#### Identifying Relationships")
    try:
        show_img("outputs/eda/relationships.png")

    except:
        st.info("Run `python train_models.py` first.")

# ---------- TAB 3: Model Results ----------
with tab_results:
    st.markdown("### 📈 Model Performance Comparison")
    try:
        show_img("outputs/results/model_comparison.png")
    except:
        st.info("Run `python train_models.py` first.")

    # Metrics as a graph (instead of a table)


    metric_keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                   ("f1", "F1 Score"), ("auc", "AUC")]
    model_colors = ["#3bffa0", "#3ba0ff", "#ffc145"]

    fig, ax = plt.subplots(figsize=(12, 5))
    n_models = len(metrics)
    bar_w = 0.8 / n_models
    xs = np.arange(len(metric_keys))
    for i, (name, m) in enumerate(metrics.items()):
        vals = [m.get(key, 0) for key, _ in metric_keys]
        bars = ax.bar(xs + (i - (n_models - 1) / 2) * bar_w, vals, bar_w, label=name,
                      color=model_colors[i % len(model_colors)], edgecolor="black")
        for b, (key, _), v in zip(bars, metric_keys, vals):
            text = f"{v:.3f}" if key == "auc" else f"{v:.1%}"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, text,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label in metric_keys], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=n_models, frameon=False, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Confusion Matrices")
    cols = st.columns(3)
    files = ["cm_random_forest.png", "cm_svm_rbf_kernel.png", "cm_logistic_regression.png"]
    names = list(metrics.keys())
    for col, name, f in zip(cols, names, files):
        with col:
            st.markdown(f"**{name}**")
            try:
                show_img(f"outputs/results/{f}")
            except:
                st.info("Not available")

    st.markdown("### ROC Curves")
    cols = st.columns(3)
    files = ["roc_random_forest.png", "roc_svm_rbf_kernel.png", "roc_logistic_regression.png"]
    for col, name, f in zip(cols, names, files):
        with col:
            st.markdown(f"**{name}**")
            try:
                show_img(f"outputs/results/{f}")
            except:
                st.info("Not available")

    st.markdown("### Feature Importance (Random Forest)")
    try:
        show_img("outputs/results/feature_importance.png")
    except:
        st.info("Run `python train_models.py` first.")

# ---------- Frequent Pattern Mining (Apriori | FP-Growth) ----------
with tab_fpm:
    st.markdown("## 🧩 Frequent Pattern Mining")

    apriori_sections = [
        ("Support", ["outputs/apriori/01_support.png"]),
        ("Apriori Implementation using mlxtend", ["outputs/apriori/02_apriori.png"]),
        ("Association Rules", ["outputs/apriori/03_rules.png"]),
        ("Confidence", ["outputs/apriori/04_confidence.png"]),
        ("Lift", ["outputs/apriori/05_lift.png"]),
    ]
    fpgrowth_sections = [
        ("FP-Tree Construction", ["outputs/fpgrowth/01a_header_table.png",
                                  "outputs/fpgrowth/01b_fptree_beautiful.png",
                                  "outputs/fpgrowth/01c_tree_stats.png"]),
        ("FP-Growth Algorithm", ["outputs/fpgrowth/02_fpgrowth.png"]),
        ("Mining Frequent Itemsets", ["outputs/fpgrowth/03_mining.png"]),
        ("Comparison between Apriori and FP-Growth", ["outputs/fpgrowth/04_comparison.png"]),
    ]

    def show_chart_sections(sections):
        for title, paths in sections:
            st.markdown(f"#### {title}")
            for path in paths:
                try:
                    show_img(path)
                except:
                    st.info("Run `python train_models.py` first to generate this chart.")

    sub_apriori, sub_fpgrowth = st.tabs(["Apriori", "FP-Growth"])

    with sub_apriori:
        st.markdown("### 🛒 Apriori Algorithm")
        show_chart_sections(apriori_sections)

    with sub_fpgrowth:
        st.markdown("### 🌳 FP-Growth")
        show_chart_sections(fpgrowth_sections)

# ---------- Predict ----------
@st.fragment
def predict_section():
    st.markdown("## 🧠 Predict Compound Toxicity")

    input_mode = st.radio(
        "Compound source:",
        ["Choose from known compound library", "Enter a custom compound"],
        horizontal=True,
        label_visibility="collapsed",
    )

    smiles_input = None
    compound_label = None

    col_l, col_mid, col_r = st.columns([1, 4, 1])
    with col_mid:
        if input_mode == "Choose from known compound library":
            full_library = get_full_library()
            sel1, sel2 = st.columns(2)
            with sel1:
                compound_label = st.selectbox(
                    "Compound name",
                    list(full_library.keys()),
                    label_visibility="collapsed",
                    placeholder="Select Compound Name",
                    index=None,
                )
            with sel2:
                model_to_run = st.selectbox(
                    "Model",
                    list(models.keys()),
                    label_visibility="collapsed",
                    placeholder="Select Prediction Model",
                    index=None,
                )
            smiles_input = full_library[compound_label] if compound_label else None
        else:
            st.caption("Give your compound a name, and its structure code (SMILES).")
            c1, c2 = st.columns(2)
            with c1:
                compound_label = st.text_input(
                    "Compound name",
                    label_visibility="collapsed",
                    placeholder="Compound Name",
                )
            with c2:
                custom_smiles = st.text_input(
                    "SMILES",
                    label_visibility="collapsed",
                    placeholder="SMILES Structure Code",
                )
            compound_label = compound_label.strip() if compound_label else None
            smiles_input = custom_smiles.strip() if custom_smiles else None
            model_to_run = st.selectbox(
                "Model",
                list(models.keys()),
                label_visibility="collapsed",
                placeholder="Select Prediction Model",
                index=None,
            )

        run_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    if run_clicked:
        if not smiles_input:
            st.warning("Please choose or enter a compound first.")
        elif input_mode == "Enter a custom compound" and not compound_label:
            st.warning("Please give your compound a name.")
        elif not model_to_run:
            st.warning("Please choose a model to run.")
        else:
            feats, mol = featurize(smiles_input)
            if feats is None:
                st.error("Couldn't read that compound's structure code. Please check it and try again.")
            else:
                if input_mode == "Enter a custom compound":
                    saved = save_custom_compound(compound_label, smiles_input)
                    if saved:
                        st.success(f"✅ **{compound_label}** was saved to your compound library — "
                                   f"you'll find it under 'Choose from known compound library' next time.")
                    else:
                        st.info(f"ℹ️ **{compound_label}** is already in your compound library.")

                st.markdown(f"### 🧪 Screening Report: **{compound_label}**")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(Draw.MolToImage(mol, size=(240, 240)), caption=compound_label)
                with c2:
                    X = pd.DataFrame([feats])[feature_names]
                    X_scaled = scaler.transform(X)
                    model = models[model_to_run]
                    Xin = X.values if model_to_run == "Random Forest" else X_scaled
                    pred = model.predict(Xin)[0]
                    prob = model.predict_proba(Xin)[0, 1]
                    # Keep the displayed percentage consistent with the predicted label
                    # (SVM's probability calibration can occasionally disagree with its own decision boundary)
                    if pred == 1 and prob < 0.5:
                        prob = 0.5
                    elif pred == 0 and prob >= 0.5:
                        prob = 0.4999

                    # --- Toxicity Prediction ---
                    st.markdown("#### ⚠️ Toxicity Screening")
                    st.markdown(f"**Model used:** {MODEL_ICONS.get(model_to_run,'🔹')} {model_to_run}")
                    if pred == 1:
                        st.markdown(f"<span class='verdict-toxic'>⚠️ TOXIC — {prob:.0%} risk of mitochondrial toxicity</span>",
                                    unsafe_allow_html=True)
                        st.caption("❌ **Recommendation:** Compound should be flagged for further review before advancing in the drug discovery pipeline.")
                    else:
                        st.markdown(f"<span class='verdict-safe'>✅ SAFE — {prob:.0%} risk of mitochondrial toxicity</span>",
                                    unsafe_allow_html=True)
                        st.caption("✅ **Recommendation:** Compound passes initial toxicity screening.")

                # --- Drug-Likeness (Lipinski's Rule of 5) ---
                st.markdown("---")
                st.markdown("#### 💊 Drug-Likeness Assessment (Lipinski's Rule of 5)")
                rules, violations = check_drug_likeness(feats)

                lc1, lc2 = st.columns([2, 1])
                with lc1:
                    for rule, (passed, value) in rules.items():
                        icon = "✅" if passed else "❌"
                        color = "var(--accent)" if passed else "var(--danger)"
                        st.markdown(
                            f"<div style='padding:6px 0; color:{color};'>"
                            f"{icon} <b>{rule}</b> — <span style='color:var(--text-secondary);'>Actual: {value}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                with lc2:
                    if violations == 0:
                        verdict = "🌟 Excellent Drug Candidate"
                        color_class = "verdict-safe"
                        note = "Passes all 4 rules — ideal for oral bioavailability."
                    elif violations == 1:
                        verdict = "✅ Drug-Like"
                        color_class = "verdict-safe"
                        note = "1 violation is acceptable in modern drug discovery."
                    else:
                        verdict = f"⚠️ {violations} Violations"
                        color_class = "verdict-toxic"
                        note = "Poor drug-likeness — likely oral bioavailability issues."
                    st.markdown(f"<span class='{color_class}'>{verdict}</span>", unsafe_allow_html=True)
                    st.caption(note)

                # --- Overall Drug Discovery Verdict ---
                st.markdown("---")
                st.markdown("#### 🎯 Overall Pipeline Recommendation")
                if pred == 0 and violations <= 1:
                    st.success("🟢 **ADVANCE** — Compound passes both toxicity screening and drug-likeness. "
                               "Recommended for next stage of drug discovery pipeline (bioactivity testing).")
                elif pred == 0 and violations >= 2:
                    st.warning("🟡 **CAUTION** — Non-toxic but poor drug-likeness. "
                               "May require structural optimization before advancing.")
                elif pred == 1 and violations <= 1:
                    st.warning("🟡 **REJECT (Toxicity)** — Good drug-likeness but toxicity concerns. "
                               "Consider structural modifications to reduce toxicity.")
                else:
                    st.error("🔴 **REJECT** — Both toxicity and drug-likeness concerns. "
                             "Not recommended for further development.")


with tab_predict:
    predict_section()