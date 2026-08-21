# 💊 Drug Discovery Prediction

**A Data Mining Approach to Predicting Compound Toxicity**

An early-stage drug discovery screening tool that predicts compound toxicity and assesses drug-likeness from molecular structure (SMILES), covering the **ADMET screening** stage of the drug discovery pipeline.

---

## 📋 Project Information

This project applies a full data mining pipeline — data cleaning, feature extraction, exploratory data analysis, classification, and clustering — to a real-world dataset (NIH/EPA Tox21). SMILES structures are converted into 12 molecular descriptors using RDKit, then three classification models are trained to predict mitochondrial toxicity. Compounds are additionally grouped by structural similarity using K-Means clustering, and assessed for drug-likeness using Lipinski's Rule of Five.

**Domain application:** Toxicity & drug-likeness screening — an early filtering stage in drug discovery (ADMET screening).

---

## 🧬 Dataset

- **Source:** Tox21 (NIH/EPA) — a real-world chemical toxicity dataset
- **Target:** `SR-MMP` (Stress Response — Mitochondrial Membrane Potential disruption)
- **Input:** SMILES strings → 12 molecular descriptors extracted with RDKit
- **Task:** Binary classification (Toxic = 1, Non-toxic = 0)

---

## 🔬 Data Mining Pipeline

| Step | What it does |
|------|---------------|
| **1. Load Data** | Reads `Dataset/tox21.csv` |
| **2. Preprocessing** | Handles missing values, removes duplicate rows, prints descriptive statistics |
| **3. Feature Extraction** | Converts each compound's SMILES into 12 molecular descriptors (MolWt, LogP, TPSA, H-bond donors/acceptors, aromatic rings, etc.) using RDKit |
| **4. EDA** | Class distribution, feature distributions, correlation heatmap, feature-vs-toxicity boxplots |
| **5. Train/Test Split** | 80/20 split with stratification, feature scaling via `StandardScaler` |
| **6. Classification** | Trains **Random Forest**, **SVM (RBF kernel)**, and **Logistic Regression** — evaluated with accuracy, precision, recall, F1, AUC, confusion matrices, ROC curves |
| **7. Clustering** | **K-Means** clustering on molecular descriptors — elbow method to choose *k*, PCA 2D visualization, per-cluster interpretation (average descriptor values + toxicity rate) |
| **8. Save Artifacts** | Trained models, scaler, and metrics saved for use in the Streamlit app |

---

## 🤖 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 88% |
| SVM (RBF kernel) | 80% |
| Logistic Regression | 77% |

---

## 💻 App Features (Streamlit)

The app has 5 tabs:

- **🗃️ Dataset** — raw data overview, missing values
- **📊 EDA** — all exploratory charts
- **📈 Model Results** — accuracy comparison, confusion matrices, ROC curves, feature importance
- **🔗 Clustering** — K-Means elbow curve, cluster visualization, cluster interpretation table
- **🤖 Predict** — run a live prediction on any compound

### Prediction workflow
For any compound (from the built-in library, your saved custom compounds, or a freshly typed name + SMILES), the app runs:

1. **⚠️ Toxicity Screening** — ML-predicted risk of mitochondrial toxicity (choose which of the 3 models to use)
2. **💊 Drug-Likeness Assessment** — Lipinski's Rule of Five (Molecular Weight ≤ 500 Da, LogP ≤ 5, H-bond Donors ≤ 5, H-bond Acceptors ≤ 10)
3. **🎯 Overall Pipeline Recommendation** — combines both results into **ADVANCE / CAUTION / REJECT**

### Custom compound library
Compounds entered manually are validated (via RDKit) and saved to `Dataset/custom_compounds.csv`, so they appear in the "known compound library" dropdown on future runs.

---

## 📁 Project Structure

```
files/
├── Dataset/
│   ├── tox21.csv                 # Real Tox21 dataset
│   └── custom_compounds.csv      # User-saved compounds (Name, SMILES)
├── models/
│   ├── models.pkl                # Trained models + scaler + K-Means + PCA
│   └── metrics.json              # Model performance metrics
├── outputs/
│   ├── eda/                      # EDA charts
│   └── results/                  # Confusion matrices, ROC curves, clustering charts
├── train_models.py               # Full data mining + ML training pipeline
├── app.py                        # Streamlit web interface
└── requirements.txt
```

---

## ⚙️ Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the models** (generates `models/`, `outputs/eda/`, `outputs/results/`)
```bash
python train_models.py
```

**3. Launch the app**
```bash
streamlit run app.py
```

---

## 🔍 Finding Compound Names & SMILES

To test the app with your own compounds:

- **[PubChem](https://pubchem.ncbi.nlm.nih.gov/)** — search any compound by name, copy its *Canonical SMILES*
- **[Top 200 Drugs List](https://litfl.com/bscc/top-200-drugs-pharmacology/)** — browse well-known drug names to look up

---

## 🛠️ Tech Stack

- **Cheminformatics:** RDKit
- **Machine Learning:** scikit-learn (Random Forest, SVM, Logistic Regression, K-Means, PCA)
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit

---

## ⚠️ Scope

This project covers the **ADMET screening** stage of drug discovery (toxicity + drug-likeness filtering) — it does not perform target identification, hit discovery, lead optimization, or clinical trial stages.

---

## 👤 Author

**Rakibun Ateed**

- GitHub: [@rakibunateed](https://github.com/rakibunateed)