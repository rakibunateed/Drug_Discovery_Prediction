# 💊 Drug Discovery Prediction

**A Data Mining Approach to Predicting Compound Toxicity**

An early-stage drug discovery screening tool that predicts compound toxicity and assesses drug-likeness from molecular structure (SMILES) and assay results, covering the **ADMET screening** stage of the drug discovery pipeline.

---

## 📋 Project Information

This project applies a full data mining pipeline — data cleaning, feature extraction, exploratory data analysis, classification, clustering, and frequent pattern mining — to a real-world dataset (NIH/EPA Tox21).

Each compound's SMILES structure is converted into **12 molecular descriptors** with RDKit. These are combined with the dataset's **other 11 assay results**, and three classification models are trained to predict mitochondrial toxicity. Compounds are also assessed for drug-likeness with Lipinski's Rule of Five.

**Domain application:** Toxicity & drug-likeness screening — an early filtering stage in drug discovery (ADMET screening).

---

## 🧬 Dataset

- **Source:** Tox21 (NIH/EPA) — a real-world chemical toxicity dataset
- **File:** `Dataset/tox21.csv` (`mol_id`, `smiles`, and 12 assay columns)
- **Target:** `SR-MMP` (Stress Response — Mitochondrial Membrane Potential disruption)
- **Task:** Binary classification (Toxic = 1, Non-toxic = 0)

### Model inputs (34 features)

| Input | Count | Description |
|-------|-------|-------------|
| Molecular descriptors | 12 | Calculated from the SMILES with RDKit: MolWt, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic rings, ring count, heavy atoms, Fraction Csp3, valence electrons, molar refractivity |
| Assay results | 22 | The 11 other assays (everything except `SR-MMP`), each encoded as two columns: `<assay>_toxic` (1 = Toxic) and `<assay>_not_measured` (1 = missing in the dataset). Both 0 means Non-toxic |

`SR-MMP` is the target, so it is **never** used as an input.

---

## 🔬 Data Mining Pipeline

| Step | What it does |
|------|---------------|
| **1. Load Data** | Reads `Dataset/tox21.csv` |
| **2. Preprocessing** | Handles missing values, removes duplicate rows, prints descriptive statistics |
| **3. Feature Extraction** | Converts each SMILES into 12 RDKit descriptors and adds the 22 assay columns |
| **4. EDA** | Histograms, bar chart, scatter plot, box plot, correlation heatmap, outliers, relationships |
| **5. Train/Test Split** | 80/20 stratified split (`random_state=42`), feature scaling via `StandardScaler` |
| **6. Classification** | Trains **Random Forest**, **SVM (RBF kernel)**, and **Logistic Regression** — evaluated with accuracy, precision, recall, F1, AUC, confusion matrices, ROC curves, and feature importance |
| **7. Clustering** | **K-Means** with the elbow method to choose *k*, PCA 2D visualization, per-cluster summary |
| **8. Save Artifacts** | Trained models, scaler, and metrics saved for the Streamlit app |
| **9. Apriori** | Association rules on the Tox21 assays: support, confidence, lift |
| **10. FP-Growth** | FP-tree construction, mining frequent itemsets, comparison with Apriori |

---

## 🤖 Model Performance

Run `python train_models.py`, then open the **📈 Model Results** tab in the app to see the final accuracy, precision, recall, F1 and AUC for each model.

| Model                | Accuracy |
|----------------------|----------|
| Random Forest        | *92%*    |
| SVM (RBF kernel)     | *89%*    |
| Logistic Regression  | *87%*    |

---

## 💻 App Features (Streamlit)

The app has 5 tabs:

- **🗃️ Dataset** — raw data overview and missing values
- **📊 EDA** — all exploratory charts
- **🧩 Frequent Pattern Mining** — Apriori and FP-Growth results
- **📈 Model Results** — accuracy comparison, confusion matrices, ROC curves, feature importance
- **🤖 Predict** — predict toxicity for a compound from the dataset

### Prediction workflow

Predictions use **only compounds from the Tox21 dataset** (`Dataset/tox21.csv`).

1. **Select a compound** from the dataset. The 11 assay boxes, `mol_id` and `smiles` fill in automatically.
2. *(Optional)* **Edit any assay box** (Toxic / Non-toxic / Not measured). The models use these values as inputs, so changing them can change the prediction.
3. **Select a model** (Random Forest, SVM, or Logistic Regression) and click **Run Prediction**.

The screening report then shows:

- **⚠️ Toxicity Screening** — predicted risk of mitochondrial toxicity, compared with the real `SR-MMP` result from the dataset (skipped if the assay boxes were edited)
- **💊 Drug-Likeness Assessment** — Lipinski's Rule of Five (Molecular Weight ≤ 500 Da, LogP ≤ 5, H-bond Donors ≤ 5, H-bond Acceptors ≤ 10)
- **🎯 Overall Pipeline Recommendation** — combines both results into **ADVANCE / CAUTION / REJECT**

The app rebuilds the same 80/20 split used in training, so it can tell you whether a compound was in the **training set** (the model has seen it) or the **test set** (the model has never seen it).

---

## 📁 Project Structure

```
files/
├── Dataset/
│   └── tox21.csv                 # Real Tox21 dataset (source of all predictions)
├── models/
│   ├── models.pkl                # Trained models + scaler + feature names + K-Means + PCA
│   └── metrics.json              # Model performance metrics
├── outputs/
│   ├── eda/                      # EDA charts
│   ├── results/                  # Confusion matrices, ROC curves, K-Means charts
│   ├── apriori/                  # Apriori charts
│   └── fpgrowth/                 # FP-Growth charts
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

**2. Train the models** (generates `models/` and everything in `outputs/`)
```bash
python train_models.py
```

**3. Launch the app**
```bash
streamlit run app.py
```

> If you change the model inputs in `train_models.py`, run step 2 again. The app and the saved models must use the same feature names.

---

## 🛠️ Tech Stack

- **Cheminformatics:** RDKit
- **Machine Learning:** scikit-learn (Random Forest, SVM, Logistic Regression, K-Means, PCA)
- **Pattern Mining:** mlxtend (Apriori, FP-Growth)
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit (version 1.35 or newer)

---

## ⚠️ Scope

This project covers the **ADMET screening** stage of drug discovery (toxicity + drug-likeness filtering). It does not perform target identification, hit discovery, lead optimization, or clinical trial stages.

In real early screening, the other assay results may not be available before a compound is tested. Here they are used as extra inputs because they are part of the Tox21 dataset.

---

## 👤 Author

**Rakibun Ateed**

- GitHub: [@rakibunateed](https://github.com/rakibunateed)