"""
train_models.py — Tox21 Compound Toxicity Model Training
Loads tox21.csv, extracts molecular features from SMILES,
trains 3 ML models, saves models + metrics + EDA charts.
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# ============================================================
# 0. Setup output folders
# ============================================================
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/eda", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'

# ============================================================
# 1. LOAD REAL DATASET (Tox21 from NIH/EPA - Kaggle)
# ============================================================
print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv("Dataset/tox21.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDescriptive statistics:")
print(df.describe())

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

TARGET = "SR-MMP"  # Mitochondrial Membrane Potential toxicity

print(f"\nMissing values per column (top 5):")
print(df.isnull().sum().head())

# Check and remove duplicate rows
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows found: {dup_count}")
df = df.drop_duplicates()
print(f"Rows after removing duplicates: {len(df)}")

# Drop rows where target or SMILES is missing
before = len(df)
df = df.dropna(subset=[TARGET, "smiles"])
after = len(df)
print(f"\nRows before cleaning: {before}")
print(f"Rows after dropping missing {TARGET}/smiles: {after}")
print(f"Rows removed: {before - after}")

df[TARGET] = df[TARGET].astype(int)

# ============================================================
# 3. FEATURE EXTRACTION FROM SMILES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE EXTRACTION (SMILES -> Molecular Descriptors)")
print("=" * 60)

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
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

print("Extracting features (this may take a minute)...")
features_list = []
labels_list = []
for _, row in df.iterrows():
    feats = extract_features(row["smiles"])
    if feats is not None:
        features_list.append(feats)
        labels_list.append(row[TARGET])

X_df = pd.DataFrame(features_list)
y = np.array(labels_list)
print(f"Successfully featurized {len(X_df)} compounds")
print(f"Features: {list(X_df.columns)}")

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 4.1 Class balance
plt.figure(figsize=(6, 4))
counts = pd.Series(y).value_counts()
sns.barplot(x=["Non-toxic (0)", "Toxic (1)"], y=counts.values, palette=["#3bffa0", "#ff5c5c"])
plt.title(f"Class Distribution — {TARGET}")
plt.ylabel("Count")
for i, v in enumerate(counts.values):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda/class_distribution.png", dpi=100)
plt.close()
print(f"[Saved] outputs/eda/class_distribution.png")
print(f"Class balance: Non-toxic={counts.get(0,0)}, Toxic={counts.get(1,0)}")

# 4.2 Feature distributions
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flatten(), X_df.columns):
    ax.hist(X_df[col], bins=30, color="#3bffa0", edgecolor="black", alpha=0.8)
    ax.set_title(col, fontsize=10)
plt.suptitle("Feature Distributions", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda/feature_distributions.png", dpi=100)
plt.close()
print(f"[Saved] outputs/eda/feature_distributions.png")

# 4.3 Correlation heatmap
plt.figure(figsize=(10, 8))
corr = X_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", dpi=100)
plt.close()
print(f"[Saved] outputs/eda/correlation_heatmap.png")

# 4.4 Feature vs Toxicity (boxplots)
key_features = ["MolWt", "LogP", "TPSA", "NumAromaticRings"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
plot_df = X_df.copy()
plot_df["Toxicity"] = y
for ax, feat in zip(axes, key_features):
    sns.boxplot(x="Toxicity", y=feat, data=plot_df, ax=ax, palette=["#3bffa0", "#ff5c5c"])
    ax.set_xticklabels(["Non-toxic", "Toxic"])
    ax.set_title(f"{feat} by Toxicity")
plt.tight_layout()
plt.savefig("outputs/eda/feature_vs_toxicity.png", dpi=100)
plt.close()
print(f"[Saved] outputs/eda/feature_vs_toxicity.png")

# ============================================================
# 5. TRAIN/TEST SPLIT + SCALING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: TRAIN/TEST SPLIT")
print("=" * 60)

feature_names = list(X_df.columns)
X = X_df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. DATA MINING — TRAIN 3 ML MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: TRAINING MODELS")
print("=" * 60)

models_to_train = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
}

trained_models = {}
metrics = {}

for name, model in models_to_train.items():
    print(f"\n--- {name} ---")
    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save confusion matrix chart
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Non-toxic", "Toxic"],
                yticklabels=["Non-toxic", "Toxic"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"outputs/results/cm_{safe}.png", dpi=100)
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#3bffa0", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"outputs/results/roc_{safe}.png", dpi=100)
    plt.close()

    trained_models[name] = model
    metrics[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }

# 6.1 Feature importance (Random Forest)
rf = trained_models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#3bffa0", edgecolor="black")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/results/feature_importance.png", dpi=100)
plt.close()
print("\n[Saved] outputs/results/feature_importance.png")

# 6.2 Accuracy comparison chart
plt.figure(figsize=(7, 4))
names = list(metrics.keys())
accs = [metrics[n]["accuracy"] for n in names]
sns.barplot(x=names, y=accs, palette=["#3bffa0", "#3ba0ff", "#ffc145"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for i, v in enumerate(accs):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/results/model_comparison.png", dpi=100)
plt.close()
print("[Saved] outputs/results/model_comparison.png")

# ============================================================
# 7. DATA MINING — K-MEANS CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: K-MEANS CLUSTERING")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Use the same scaled features as SVM/Logistic Regression
X_all_scaled = scaler.transform(X)

# 7.1 Elbow method — find a good number of clusters
print("\nRunning elbow method (k = 1 to 10)...")
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_all_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(k_range), inertias, marker="o", color="#3bffa0")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("outputs/results/kmeans_elbow.png", dpi=100)
plt.close()
print("[Saved] outputs/results/kmeans_elbow.png")

# 7.2 Fit K-Means with chosen k
K_CHOSEN = 3
print(f"\nFitting K-Means with k={K_CHOSEN} (chosen from elbow curve)...")
kmeans = KMeans(n_clusters=K_CHOSEN, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_all_scaled)
print(f"Cluster sizes: {pd.Series(cluster_labels).value_counts().sort_index().to_dict()}")

# 7.3 Visualize clusters in 2D using PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all_scaled)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6, s=15)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"K-Means Clusters (k={K_CHOSEN}) — PCA Projection")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig("outputs/results/kmeans_clusters.png", dpi=100)
plt.close()
print("[Saved] outputs/results/kmeans_clusters.png")

# 7.4 Interpret clusters — average feature values + toxicity rate per cluster
cluster_df = X_df.copy()
cluster_df["Cluster"] = cluster_labels
cluster_df["Toxicity"] = y
cluster_summary = cluster_df.groupby("Cluster").mean(numeric_only=True).round(2)
cluster_summary["Compound Count"] = cluster_df.groupby("Cluster").size()
print("\nCluster summary (mean feature values + toxicity rate per cluster):")
print(cluster_summary)
cluster_summary.to_csv("outputs/results/kmeans_cluster_summary.csv")
print("[Saved] outputs/results/kmeans_cluster_summary.csv")

# ============================================================
# 8. SAVE ARTIFACTS FOR STREAMLIT APP
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: SAVING MODELS + METRICS")
print("=" * 60)

artifacts = {
    "models": trained_models,
    "scaler": scaler,
    "feature_names": feature_names,
    "kmeans": kmeans,
    "pca": pca,
}
with open("models/models.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("[Saved] models/models.pkl")

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[Saved] models/metrics.json")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nFinal Results:")
for name, m in metrics.items():
    print(f"  {name:22s} — Acc: {m['accuracy']:.2%}  F1: {m['f1']:.2%}  AUC: {m['auc']:.3f}")