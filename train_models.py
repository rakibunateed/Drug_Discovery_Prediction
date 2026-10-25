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

# EDA works on the 12 Tox21 toxicity labels (NR-* and SR-*).
# Missing labels are filled with 0 here ONLY for these charts —
# model training (steps 2, 3, 5, 6) is not affected.
eda_df = pd.read_csv("Dataset/tox21.csv")
numeric_cols = eda_df.select_dtypes(include=[np.number]).columns
eda_df[numeric_cols] = eda_df[numeric_cols].fillna(0)
label_cols = [c for c in eda_df.columns if c.startswith('NR-') or c.startswith('SR-')]
print(f"EDA labels ({len(label_cols)}): {label_cols}")

# 4.1 Histograms
n_cols = 4
n_rows = int(np.ceil(len(label_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10), squeeze=False)
for i, col in enumerate(label_cols):
    ax = axes[i // n_cols, i % n_cols]
    eda_df[col].hist(ax=ax, bins=3, color='skyblue', edgecolor='black')
    ax.set_title(f'{col}')
for j in range(len(label_cols), n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis('off')
plt.tight_layout()
plt.savefig("outputs/eda/histograms.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/histograms.png")

# 4.2 Bar chart
toxic_counts = eda_df[label_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
toxic_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Number of Toxic Compounds per Target')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/eda/bar_chart.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/bar_chart.png")

# 4.3 Scatter plot
np.random.seed(42)
plt.figure(figsize=(8, 6))
plt.scatter(eda_df[label_cols[0]] + np.random.normal(0, 0.05, len(eda_df)),
            eda_df[label_cols[1]] + np.random.normal(0, 0.05, len(eda_df)),
            alpha=0.3, c='purple')
plt.xlabel(label_cols[0])
plt.ylabel(label_cols[1])
plt.title(f'Scatter: {label_cols[0]} vs {label_cols[1]}')
plt.tight_layout()
plt.savefig("outputs/eda/scatter_plot.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/scatter_plot.png")

# 4.4 Box plot
plt.figure(figsize=(12, 5))
eda_df[label_cols].boxplot()
plt.title('Boxplot of Toxicity Labels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/eda/box_plot.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/box_plot.png")

# 4.5 Correlation matrix + heatmap
plt.figure(figsize=(10, 8))
corr = eda_df[label_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap of Toxicity Targets')
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/correlation_heatmap.png")

# 4.6 Identifying outliers (printed ratios + graphical representation)
print("\nOutlier analysis (toxic compound ratio):")
for col in label_cols:
    ratio = eda_df[col].mean() * 100
    print(f"{col}: {ratio:.2f}% toxic")

# IQR rule: a value is an outlier if it lies outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
def iqr_fences(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

# (a) Targets: which assays have an unusually high / low toxic ratio?
toxic_ratio = (eda_df[label_cols].mean() * 100).sort_values(ascending=False)
low_fence, high_fence = iqr_fences(toxic_ratio.values)
target_outliers = toxic_ratio[(toxic_ratio > high_fence) | (toxic_ratio < low_fence)]

# (b) Compounds: how many targets is each compound toxic in?
toxic_per_compound = eda_df[label_cols].sum(axis=1).astype(int)
_, comp_high_fence = iqr_fences(toxic_per_compound.values)
compound_outliers = int((toxic_per_compound > comp_high_fence).sum())
print(f"\nTargets flagged as outliers (IQR rule): {list(target_outliers.index) or 'none'}")
print(f"Compounds toxic in more than {comp_high_fence:.1f} targets (outliers): {compound_outliers}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

bar_colors = ['crimson' if name in target_outliers.index else 'skyblue' for name in toxic_ratio.index]
ax1.bar(toxic_ratio.index, toxic_ratio.values, color=bar_colors, edgecolor='black')
ax1.axhline(toxic_ratio.mean(), color='gray', linestyle='--', label=f'Mean ({toxic_ratio.mean():.1f}%)')
ax1.axhline(high_fence, color='crimson', linestyle=':', label=f'Upper fence ({high_fence:.1f}%)')
ax1.set_title('Outliers by Target: Toxic Compound Ratio')
ax1.set_ylabel('Toxic compounds (%)')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.legend()

count_dist = toxic_per_compound.value_counts().sort_index()
count_colors = ['crimson' if k > comp_high_fence else 'skyblue' for k in count_dist.index]
ax2.bar(count_dist.index, count_dist.values, color=count_colors, edgecolor='black')
ax2.axvline(comp_high_fence, color='crimson', linestyle=':',
            label=f'Outlier fence ({comp_high_fence:.1f}) — {compound_outliers} compounds flagged')
ax2.set_yscale('log')
ax2.set_xticks(count_dist.index)
ax2.set_title('Outliers by Compound: Number of Targets Each Compound is Toxic In')
ax2.set_xlabel('Number of toxic targets')
ax2.set_ylabel('Number of compounds (log scale)')
ax2.legend()

plt.tight_layout()
plt.savefig("outputs/eda/outliers.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/outliers.png")

# 4.7 Identifying relationships — strongest correlations between targets
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
top_pairs = pairs.sort_values(ascending=False).head(10)
pair_labels = [f"{a} vs {b}" for a, b in top_pairs.index]

plt.figure(figsize=(10, 5.5))
plt.barh(pair_labels[::-1], top_pairs.values[::-1], color='coral', edgecolor='black')
plt.xlabel('Correlation')
plt.title('Top 10 Strongest Relationships Between Toxicity Targets')
for idx, v in enumerate(top_pairs.values[::-1]):
    plt.text(v + 0.005, idx, f"{v:.2f}", va='center')
plt.xlim(0, max(top_pairs.values) + 0.1)
plt.tight_layout()
plt.savefig("outputs/eda/relationships.png", dpi=100)
plt.close()
print("[Saved] outputs/eda/relationships.png")
print("\nStrongest relationships:")
print(top_pairs.round(2).to_string())

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

# ============================================================
# 9. ASSOCIATION RULE MINING — APRIORI ALGORITHM
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: APRIORI ALGORITHM")
print("=" * 60)

def run_apriori_analysis():
    """Apriori / association rules on the Tox21 assays. Saves 5 charts to outputs/apriori/."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        print("⚠️  mlxtend is not installed, so the Apriori step was skipped.")
        print("    Install it with:  pip install mlxtend   and run this script again.")
        return

    OUT = "outputs/apriori"
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv('Dataset/tox21.csv')

    # Prepare transaction data
    tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                   'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                   'SR-HSE', 'SR-MMP', 'SR-p53']

    df_trans = (df[tox21_tasks] == 1).fillna(False)
    df_trans.columns = [c + '_Toxic' for c in tox21_tasks]
    df_trans = df_trans[df_trans.any(axis=1)]

    print("✅ Setup complete")
    print(f"Dataset: {df.shape[0]} compounds")
    print(f"Transactions: {df_trans.shape[0]}")
    print(f"Items: {df_trans.shape[1]}")
    print(df_trans.head())


    print("=" * 70)
    print("SUPPORT")
    print("=" * 70)
    print("""
DEFINITION:
Support tells us how often an item/itemset appears in the dataset.

FORMULA:
Support(A) = Count(A) / Total Transactions

EXAMPLE:
If 400 out of 2000 compounds are toxic in NR-AR:
Support = 400/2000 = 0.20 (20%)
""")

    # Calculate support for each item
    support_values = df_trans.mean().sort_values(ascending=False)

    # Display table
    print(f"{'Assay':<25}{'Support':>12}{'Count':>10}")
    print("-" * 47)
    for item, sup in support_values.items():
        count = int(round(sup * len(df_trans)))
        print(f"{item:<25}{sup:>12.4f}{count:>10}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of support
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(support_values)))
    axes[0].barh(support_values.index, support_values.values,
                 color=colors, edgecolor='black')
    axes[0].set_xlabel('Support', fontsize=12, fontweight='bold')
    axes[0].set_title('Support of Each Toxicity Assay', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    for i, v in enumerate(support_values.values):
        axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)

    # Pie chart – proportion of support
    axes[1].pie(support_values.values, labels=support_values.index,
                autopct='%1.1f%%', startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(support_values))))
    axes[1].set_title('Support Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/01_support.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/01_support.png")


    print("=" * 70)
    print("APRIORI IMPLEMENTATION USING MLXTEND")
    print("=" * 70)
    print("""
APRIORI ALGORITHM STEPS:
1. Find frequent 1-itemsets (items with min support)
2. Generate candidate 2-itemsets from frequent 1-itemsets
3. Prune using Apriori property
4. Repeat until no more frequent itemsets

APRIORI PROPERTY:
All subsets of a frequent itemset must also be frequent.
""")

    # Apply Apriori
    min_support = 0.01
    frequent_itemsets = apriori(df_trans, min_support=min_support,
                                use_colnames=True, max_len=4)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

    print(f"Minimum Support: {min_support}")
    print(f"Frequent Itemsets Found: {len(frequent_itemsets)}\n")

    # Group by size
    size_counts = frequent_itemsets['length'].value_counts().sort_index()
    print("Itemsets grouped by size:")
    for size, count in size_counts.items():
        print(f"  Size {size}: {count} itemsets")

    # Top 10
    print("\n--- Top 10 Frequent Itemsets ---")
    top10 = frequent_itemsets.sort_values('support', ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"  Support={row['support']:.4f} | {set(row['itemsets'])}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Itemsets by size
    axes[0].bar(size_counts.index.astype(str), size_counts.values,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(size_counts)],
                edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Itemset Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Frequent Itemsets', fontsize=12, fontweight='bold')
    axes[0].set_title('Frequent Itemsets by Size', fontsize=14, fontweight='bold')
    for i, v in enumerate(size_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Chart 2: Top 10 itemsets by support
    top10_labels = [', '.join(list(x))[:30] for x in top10['itemsets']]
    axes[1].barh(range(len(top10)), top10['support'].values,
                 color=plt.cm.plasma(np.linspace(0.2, 0.9, len(top10))),
                 edgecolor='black')
    axes[1].set_yticks(range(len(top10)))
    axes[1].set_yticklabels(top10_labels, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Support', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 10 Frequent Itemsets', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/02_apriori.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/02_apriori.png")


    print("=" * 70)
    print("ASSOCIATION RULES")
    print("=" * 70)
    print("""
DEFINITION:
An association rule is of the form A → B
Meaning: "If A occurs, then B is likely to occur"

A = Antecedent (Left side)
B = Consequent (Right side)

EXAMPLE:
Rule: {NR-AR_Toxic} → {NR-AR-LBD_Toxic}
→ "Compounds toxic in NR-AR are also likely toxic in NR-AR-LBD"
""")

    # Generate rules (newer mlxtend versions also want the number of transactions)
    try:
        rules = association_rules(frequent_itemsets, num_itemsets=len(df_trans),
                                  metric="confidence", min_threshold=0.3)
    except TypeError:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

    rules['antecedent'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequent'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['rule'] = rules['antecedent'] + ' → ' + rules['consequent']

    print(f"Total Association Rules Generated: {len(rules)}\n")
    print("Sample Rules:")
    print(rules[['antecedent', 'consequent', 'support', 'confidence', 'lift']].head(10).to_string(index=False))

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Network-like scatter (rules space)
    sc = axes[0].scatter(rules['support'], rules['confidence'],
                         c=rules['lift'], s=100, cmap='YlOrRd',
                         edgecolors='black', alpha=0.8)
    axes[0].set_xlabel('Support', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Confidence', fontsize=12, fontweight='bold')
    axes[0].set_title('Association Rules Space', fontsize=14, fontweight='bold')
    plt.colorbar(sc, ax=axes[0], label='Lift')
    axes[0].grid(alpha=0.3)

    # Chart 2: Top 10 rules
    top10_rules = rules.sort_values('lift', ascending=False).head(10)
    labels = [r[:40] + '...' if len(r) > 40 else r for r in top10_rules['rule']]
    axes[1].barh(range(len(top10_rules)), top10_rules['lift'].values,
                 color=plt.cm.coolwarm(np.linspace(0.2, 0.9, len(top10_rules))),
                 edgecolor='black')
    axes[1].set_yticks(range(len(top10_rules)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Lift', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 10 Rules (by Lift)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/03_rules.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/03_rules.png")

    print("=" * 70)
    print("CONFIDENCE")
    print("=" * 70)
    print("""
DEFINITION:
Confidence measures how often the rule A → B is true.
Probability of B given A.

FORMULA:
Confidence(A → B) = Support(A ∪ B) / Support(A)

EXAMPLE:
If 300 compounds toxic in NR-AR, and 240 also toxic in NR-AR-LBD:
Confidence = 240/300 = 0.80 (80%)

INTERPRETATION:
High Confidence → Strong Rule
Low Confidence → Weak Rule
""")

    top_conf = rules.sort_values('confidence', ascending=False).head(15)

    print("--- Top 15 Rules by CONFIDENCE ---\n")
    print(f"{'Rule':<70}{'Confidence':>12}")
    print("-" * 82)
    for _, r in top_conf.iterrows():
        rule_str = f"{r['antecedent']} → {r['consequent']}"
        if len(rule_str) > 68:
            rule_str = rule_str[:65] + '...'
        print(f"{rule_str:<70}{r['confidence']:>12.4f}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Histogram of confidence
    axes[0].hist(rules['confidence'], bins=25, color='#2E86AB',
                 edgecolor='black', alpha=0.8)
    axes[0].axvline(rules['confidence'].mean(), color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean = {rules['confidence'].mean():.3f}")
    axes[0].set_xlabel('Confidence', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Confidence Values', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Chart 2: Top 10 by confidence
    top10_conf = rules.sort_values('confidence', ascending=False).head(10)
    labels = [f"{r['antecedent']} → {r['consequent']}"[:40] for _, r in top10_conf.iterrows()]
    bars = axes[1].barh(range(len(top10_conf)), top10_conf['confidence'].values,
                        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top10_conf))),
                        edgecolor='black')
    axes[1].set_yticks(range(len(top10_conf)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Confidence', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 10 Rules by Confidence', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 1.05)
    for i, v in enumerate(top10_conf['confidence'].values):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/04_confidence.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/04_confidence.png")

    print("=" * 70)
    print("LIFT")
    print("=" * 70)
    print("""
DEFINITION:
Lift measures how much more likely A and B occur together
compared to being independent.

FORMULA:
Lift(A → B) = Confidence(A → B) / Support(B)

INTERPRETATION:
Lift > 1 → Positive correlation (occur together more than random)
Lift = 1 → Independent (no relationship)
Lift < 1 → Negative correlation (occur together less than random)

EXAMPLE:
Lift = 3.5 means the rule is 3.5× more likely than random chance.
""")

    top_lift = rules.sort_values('lift', ascending=False).head(15)

    print("--- Top 15 Rules by LIFT ---\n")
    print(f"{'Rule':<65}{'Support':>10}{'Conf':>8}{'Lift':>8}")
    print("-" * 91)
    for _, r in top_lift.iterrows():
        rule_str = f"{r['antecedent']} → {r['consequent']}"
        if len(rule_str) > 63:
            rule_str = rule_str[:60] + '...'
        print(f"{rule_str:<65}{r['support']:>10.4f}{r['confidence']:>8.4f}{r['lift']:>8.3f}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Distribution of Lift
    axes[0].hist(rules['lift'], bins=25, color='#E63946',
                 edgecolor='black', alpha=0.8)
    axes[0].axvline(x=1, color='blue', linestyle='--', linewidth=2, label='Lift = 1 (Independent)')
    axes[0].axvline(rules['lift'].mean(), color='green', linestyle='--',
                    linewidth=2, label=f"Mean = {rules['lift'].mean():.2f}")
    axes[0].set_xlabel('Lift', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Rules', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of Lift Values', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Chart 2: Heatmap of Lift between assays
    co_occur = pd.DataFrame(np.zeros((12, 12)),
                            index=[t + '_Toxic' for t in tox21_tasks],
                            columns=[t + '_Toxic' for t in tox21_tasks])

    for _, r in rules.iterrows():
        for a in r['antecedents']:
            for c in r['consequents']:
                if a in co_occur.index and c in co_occur.columns:
                    co_occur.loc[a, c] = max(co_occur.loc[a, c], r['lift'])

    sns.heatmap(co_occur, annot=True, fmt='.1f', cmap='YlOrRd',
                mask=(co_occur == 0), linewidths=0.5,
                cbar_kws={'label': 'Lift'}, ax=axes[1],
                annot_kws={'size': 8})
    axes[1].set_title('Lift Heatmap Between Toxicity Assays', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_lift.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/05_lift.png")

    print("\n📊 SUMMARY:")
    print(f"  Rules with Lift > 1: {(rules['lift'] > 1).sum()} (Positive correlation)")
    print(f"  Rules with Lift > 2: {(rules['lift'] > 2).sum()} (Strong correlation)")
    print(f"  Rules with Lift > 5: {(rules['lift'] > 5).sum()} (Very strong correlation)")
    print(f"  Maximum Lift found: {rules['lift'].max():.3f}")

run_apriori_analysis()

# ============================================================
# 10. FREQUENT PATTERN MINING — FP-GROWTH
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: FP-GROWTH")
print("=" * 60)


def run_fpgrowth_analysis():
    """FP-Tree + FP-Growth + Apriori comparison on the Tox21 assays. Saves charts to outputs/fpgrowth/."""
    try:
        from mlxtend.frequent_patterns import apriori, fpgrowth
    except ImportError:
        print("⚠️  mlxtend is not installed, so the FP-Growth step was skipped.")
        print("    Install it with:  pip install mlxtend   and run this script again.")
        return

    import time
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from collections import Counter, defaultdict

    OUT = "outputs/fpgrowth"
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv('Dataset/tox21.csv')

    # Prepare transactions
    tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                   'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                   'SR-HSE', 'SR-MMP', 'SR-p53']

    df_trans = (df[tox21_tasks] == 1).fillna(False)
    df_trans.columns = [c + '_Toxic' for c in tox21_tasks]
    df_trans = df_trans[df_trans.any(axis=1)]

    # Convert to list of transactions (for FP-tree)
    transactions = []
    for _, row in df_trans.iterrows():
        items = [col for col in df_trans.columns if row[col]]
        if items:
            transactions.append(items)

    print("✅ Setup complete")
    print(f"Compounds: {df.shape[0]} | Transactions: {len(transactions)} | Items: {df_trans.shape[1]}")

    print("=" * 70)
    print("FP-TREE CONSTRUCTION")
    print("=" * 70)
    print("""
FP-TREE (Frequent Pattern Tree):
A compressed tree structure that stores transaction data.

CONSTRUCTION STEPS:
1. Scan database → count frequency of each item
2. Remove items below min support threshold
3. Sort items in each transaction by frequency (descending)
4. Insert sorted transactions into the tree
5. Shared prefixes are merged (compression)

BENEFITS:
✓ Compact representation of data
✓ Only 2 database scans needed
✓ No candidate generation required
""")

    # ------------------------------------------------------------
    # FP-NODE CLASS
    # ------------------------------------------------------------
    class FPNode:
        def __init__(self, item, count=1, parent=None):
            self.item = item
            self.count = count
            self.parent = parent
            self.children = {}

        def increment(self, count=1):
            self.count += count

    # ------------------------------------------------------------
    # STEP 1: Count item frequencies
    # ------------------------------------------------------------
    min_support_count = int(0.01 * len(transactions))
    item_counts = Counter()
    for trans in transactions:
        for item in trans:
            item_counts[item] += 1

    # STEP 2: Filter infrequent items
    frequent_items = {i: c for i, c in item_counts.items() if c >= min_support_count}
    sorted_items = sorted(frequent_items.items(), key=lambda x: -x[1])

    print(f"Min support count : {min_support_count}")
    print(f"Frequent items : {len(frequent_items)}\n")
    print("HEADER TABLE (sorted by frequency):")
    print(f"{'Rank':<6}{'Item':<25}{'Count':>10}")
    print("-" * 41)
    for rank, (item, cnt) in enumerate(sorted_items, 1):
        print(f"{rank:<6}{item:<25}{cnt:>10}")

    # ------------------------------------------------------------
    # STEP 3: Build FP-Tree
    # ------------------------------------------------------------
    root = FPNode('ROOT', 0)
    header_table = defaultdict(list)

    for trans in transactions:
        filtered = [i for i in trans if i in frequent_items]
        filtered.sort(key=lambda x: -frequent_items[x])
        current = root
        for item in filtered:
            if item in current.children:
                current.children[item].increment()
            else:
                node = FPNode(item, 1, current)
                current.children[item] = node
                header_table[item].append(node)
            current = current.children[item]

    def count_nodes(n):
        return 1 + sum(count_nodes(c) for c in n.children.values())

    total_nodes = count_nodes(root) - 1
    print("\n✅ FP-Tree built successfully")
    print(f"  Total nodes : {total_nodes}")
    print(f"  Total transactions : {len(transactions)}")
    print(f"  Compression ratio : {len(transactions)/total_nodes:.2f}x")

    # ============================================================
    # BEAUTIFUL VISUALIZATION
    # ============================================================

    # ---------- FIGURE 1 : HEADER TABLE ----------
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#FAFAFA')

    items_list = [i[0].replace('_Toxic', '') for i in sorted_items]
    counts_list = [i[1] for i in sorted_items]
    colors = plt.cm.turbo(np.linspace(0.15, 0.85, len(items_list)))

    bars = ax.barh(items_list, counts_list, color=colors,
                   edgecolor='white', linewidth=2, height=0.7)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency Count', fontsize=13, fontweight='bold')
    ax.set_title('FP-Tree Header Table — Item Frequencies',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_facecolor('#FAFAFA')
    ax.grid(axis='x', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, v in zip(bars, counts_list):
        ax.text(v + max(counts_list) * 0.015, bar.get_y() + bar.get_height() / 2,
                f'{v:,}', va='center', fontsize=11, fontweight='bold', color='#333')

    plt.tight_layout()
    plt.savefig(f'{OUT}/01a_header_table.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/01a_header_table.png")

    # ---------- FIGURE 2 : BEAUTIFUL FP-TREE ----------
    MAX_DEPTH = 3     # levels to draw
    TOP_CHILDREN = 3  # top-N children per node

    # 1) Collect nodes to display
    display = []  # dicts: item, count, depth, parent, x

    def collect(node, depth, parent_id):
        if depth > MAX_DEPTH:
            return
        kids = sorted(node.children.values(), key=lambda n: -n.count)[:TOP_CHILDREN]
        for k in kids:
            nid = len(display)
            display.append({'item': k.item.replace('_Toxic', ''),
                            'count': k.count, 'depth': depth,
                            'parent': parent_id, 'x': None})
            collect(k, depth + 1, nid)

    collect(root, 1, -1)

    # 2) Assign X positions (leaves get sequential slots, parents centered)
    children_of = defaultdict(list)
    for idx, d in enumerate(display):
        children_of[d['parent']].append(idx)

    counter = [0]
    LEAF_GAP = 1.6  # horizontal room per leaf node (boxes are 1.44 wide)

    def assign_x(nid):
        kids = children_of[nid]
        if not kids:
            display[nid]['x'] = counter[0]
            counter[0] += LEAF_GAP
        else:
            for k in kids:
                assign_x(k)
            display[nid]['x'] = np.mean([display[k]['x'] for k in kids])

    for r in children_of[-1]:
        assign_x(r)

    root_x = np.mean([display[r]['x'] for r in children_of[-1]])
    max_x = counter[0]

    # 3) Draw
    fig, ax = plt.subplots(figsize=(max(15, max_x * 0.75), 9))
    fig.patch.set_facecolor('#141821')
    ax.set_facecolor('#141821')

    LEVEL_COLORS = ['#00D9FF', '#FFD93D', '#FF6B9D', '#6BCB77']

    def node_pos(d):
        return d['x'], -d['depth'] * 2.2

    # --- Draw edges (curved, thickness = count) ---
    maxc = max(d['count'] for d in display)
    for idx, d in enumerate(display):
        x2, y2 = node_pos(d)
        if d['parent'] == -1:
            x1, y1 = root_x, 0
        else:
            x1, y1 = node_pos(display[d['parent']])
        lw = 0.8 + 3.5 * (d['count'] / maxc)
        arrow = FancyArrowPatch((x1, y1 - 0.42), (x2, y2 + 0.42),
                                connectionstyle="arc3,rad=0.05",
                                arrowstyle='-', linewidth=lw,
                                color=LEVEL_COLORS[(d['depth'] - 1) % 4],
                                alpha=0.45, zorder=1)
        ax.add_patch(arrow)

    # --- Draw ROOT ---
    ax.add_patch(FancyBboxPatch((root_x - 0.75, -0.42), 1.5, 0.84,
                                boxstyle="round,pad=0.06,rounding_size=0.25",
                                facecolor='#FFFFFF', edgecolor='#00D9FF',
                                linewidth=3, zorder=3))
    ax.text(root_x, 0, 'ROOT', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#141821', zorder=4)

    # --- Draw nodes ---
    for d in display:
        x, y = node_pos(d)
        c = LEVEL_COLORS[(d['depth'] - 1) % 4]
        # glow
        ax.add_patch(FancyBboxPatch((x - 0.80, y - 0.50), 1.60, 1.00,
                                    boxstyle="round,pad=0.06,rounding_size=0.25",
                                    facecolor=c, edgecolor='none', alpha=0.18, zorder=2))
        # main box
        ax.add_patch(FancyBboxPatch((x - 0.72, y - 0.44), 1.44, 0.88,
                                    boxstyle="round,pad=0.05,rounding_size=0.22",
                                    facecolor='#1E2430', edgecolor=c,
                                    linewidth=2.4, zorder=3))
        ax.text(x, y + 0.14, d['item'], ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=c, zorder=4)
        ax.text(x, y - 0.18, f"count: {d['count']}", ha='center', va='center',
                fontsize=7.5, color='#C9D1D9', zorder=4)

    # --- Level labels ---
    for lvl in range(1, MAX_DEPTH + 1):
        ax.text(-1.9, -lvl * 2.2, f'Level {lvl}', ha='right', va='center',
                fontsize=11, fontweight='bold',
                color=LEVEL_COLORS[(lvl - 1) % 4], alpha=0.85)
    ax.text(-1.9, 0, 'Root', ha='right', va='center',
            fontsize=11, fontweight='bold', color='#FFFFFF', alpha=0.85)

    ax.set_xlim(-3.2, max_x + 0.5)
    ax.set_ylim(-MAX_DEPTH * 2.2 - 1.3, 1.6)
    ax.axis('off')

    ax.set_title(f'FP-TREE STRUCTURE\nTop {TOP_CHILDREN} branches per node • {MAX_DEPTH} levels deep',
                 fontsize=17, fontweight='bold', color='white', pad=22)

    legend = [mpatches.Patch(facecolor=LEVEL_COLORS[i], label=f'Level {i+1}')
              for i in range(MAX_DEPTH)]
    ax.legend(handles=legend, loc='upper right', frameon=True,
              facecolor='#1E2430', edgecolor='#444',
              labelcolor='white', fontsize=10, ncol=MAX_DEPTH)

    ax.text(0.01, 0.02, 'Edge thickness ∝ node count',
            transform=ax.transAxes, fontsize=9.5,
            color='#8B949E', style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUT}/01b_fptree_beautiful.png', dpi=150,
                bbox_inches='tight', facecolor='#141821')
    plt.close()
    print(f"[Saved] {OUT}/01b_fptree_beautiful.png")

    # ---------- FIGURE 3 : TREE STATISTICS ----------
    def nodes_per_level(node, depth, acc):
        for c in node.children.values():
            acc[depth] = acc.get(depth, 0) + 1
            nodes_per_level(c, depth + 1, acc)

    level_stats = {}
    nodes_per_level(root, 1, level_stats)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#FAFAFA')

    lv = sorted(level_stats.keys())
    vals = [level_stats[l] for l in lv]
    bars = axes[0].bar([f'L{l}' for l in lv], vals,
                       color=plt.cm.cool(np.linspace(0.2, 0.85, len(lv))),
                       edgecolor='black', linewidth=1.5)
    axes[0].set_title('Nodes per Tree Level', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tree Level', fontweight='bold')
    axes[0].set_ylabel('Number of Nodes', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.25, linestyle='--')
    axes[0].spines[['top', 'right']].set_visible(False)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                     ha='center', fontweight='bold', fontsize=11)

    axes[1].axis('off')
    axes[1].text(0.5, 0.5,
                 f"FP-TREE SUMMARY\n\n"
                 f"Transactions : {len(transactions):,}\n"
                 f"Frequent Items : {len(frequent_items)}\n"
                 f"Total Tree Nodes : {total_nodes:,}\n"
                 f"Tree Depth : {max(lv)}\n"
                 f"Root Branches : {len(root.children)}\n"
                 f"Compression Ratio : {len(transactions)/total_nodes:.2f}x",
                 ha='center', va='center', fontsize=13, family='monospace',
                 bbox=dict(boxstyle='round,pad=1.1', facecolor='#E8F4FD',
                           edgecolor='#2E86AB', linewidth=2.5))

    plt.tight_layout()
    plt.savefig(f'{OUT}/01c_tree_stats.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/01c_tree_stats.png")

    print("=" * 70)
    print("FP-GROWTH ALGORITHM")
    print("=" * 70)
    print("""
FP-GROWTH ALGORITHM:
A divide-and-conquer approach to mine frequent itemsets
directly from the FP-tree WITHOUT candidate generation.

STEPS:
1. Construct FP-tree from the transaction database
2. For each frequent item (bottom-up in header table):
   a. Build its "conditional pattern base" (paths ending at that item)
   b. Build a "conditional FP-tree" from those paths
   c. Recursively mine the conditional FP-tree
3. Collect all frequent itemsets discovered

ADVANTAGES OVER APRIORI:
✓ No candidate generation
✓ Only 2 database scans
✓ Much faster for large datasets
""")

    # Run FP-Growth using mlxtend
    min_support = 0.01

    start = time.time()
    fp_itemsets = fpgrowth(df_trans, min_support=min_support,
                           use_colnames=True, max_len=4)
    fp_time = time.time() - start

    fp_itemsets['length'] = fp_itemsets['itemsets'].apply(len)

    print(f"Min Support: {min_support}")
    print(f"⏱️ Execution Time: {fp_time:.4f} seconds")
    print(f"📦 Frequent Itemsets Found: {len(fp_itemsets)}\n")

    print("Itemsets by size:")
    size_counts_fp = fp_itemsets['length'].value_counts().sort_index()
    for size, cnt in size_counts_fp.items():
        print(f"  Size {size}: {cnt}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Algorithm workflow
    axes[0].axis('off')
    steps = [
        "1. Scan DB → Count Item Frequencies",
        "2. Sort Items by Frequency",
        "3. Build FP-Tree",
        "4. Extract Conditional Pattern Base",
        "5. Build Conditional FP-Trees",
        "6. Mine Frequent Patterns Recursively"
    ]
    colors_flow = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77', '#4D96FF', '#9B72CB']
    for i, (step, color) in enumerate(zip(steps, colors_flow)):
        y = 5 - i
        axes[0].add_patch(plt.Rectangle((0, y - 0.3), 8, 0.6, facecolor=color,
                                        edgecolor='black', linewidth=1.5))
        axes[0].text(4, y, step, ha='center', va='center',
                     fontsize=11, fontweight='bold')
        if i < len(steps) - 1:
            axes[0].annotate('', xy=(4, y - 0.5), xytext=(4, y - 0.4),
                             arrowprops=dict(arrowstyle='->', lw=2))
    axes[0].set_xlim(-0.5, 8.5)
    axes[0].set_ylim(-1.5, 5.8)
    axes[0].set_title('FP-Growth Algorithm Workflow', fontsize=14, fontweight='bold')

    # Chart 2: Itemsets by size
    axes[1].bar(size_counts_fp.index.astype(str), size_counts_fp.values,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:len(size_counts_fp)],
                edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Itemset Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Itemsets', fontsize=12, fontweight='bold')
    axes[1].set_title('FP-Growth: Frequent Itemsets by Size', fontsize=14, fontweight='bold')
    for i, v in enumerate(size_counts_fp.values):
        axes[1].text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT}/02_fpgrowth.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/02_fpgrowth.png")

    print("=" * 70)
    print("MINING FREQUENT ITEMSETS")
    print("=" * 70)
    print("""
FREQUENT ITEMSETS:
Sets of items that appear together in at least
'min_support' fraction of transactions.

MINING PROCESS:
✓ Use FP-Growth to find all frequent itemsets
✓ Filter by different support thresholds
✓ Extract meaningful patterns
""")

    # Show top itemsets
    print("--- Top 15 Frequent Itemsets (by Support) ---\n")
    top15 = fp_itemsets.sort_values('support', ascending=False).head(15)
    print(f"{'Itemset':<70}{'Support':>10}{'Size':>6}")
    print("-" * 86)
    for _, row in top15.iterrows():
        items_str = ', '.join(list(row['itemsets']))
        if len(items_str) > 68:
            items_str = items_str[:65] + '...'
        print(f"{items_str:<70}{row['support']:>10.4f}{int(row['length']):>6}")

    # Analyze at different support thresholds
    thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    mining_results = []
    for th in thresholds:
        fi = fpgrowth(df_trans, min_support=th, use_colnames=True, max_len=4)
        mining_results.append({'min_support': th, 'itemsets': len(fi)})

    df_mining = pd.DataFrame(mining_results)
    print("\n--- Itemsets Found at Different Support Thresholds ---\n")
    print(df_mining.to_string(index=False))

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Top 10 itemsets
    top10 = fp_itemsets.sort_values('support', ascending=False).head(10)
    labels = [', '.join(list(x))[:35] for x in top10['itemsets']]
    colors_top = plt.cm.plasma(np.linspace(0.2, 0.9, len(top10)))
    axes[0].barh(range(len(top10)), top10['support'].values,
                 color=colors_top, edgecolor='black')
    axes[0].set_yticks(range(len(top10)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Support', fontsize=12, fontweight='bold')
    axes[0].set_title('Top 10 Mined Frequent Itemsets', fontsize=14, fontweight='bold')
    for i, v in enumerate(top10['support'].values):
        axes[0].text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    # Chart 2: Itemsets vs support threshold
    axes[1].plot(df_mining['min_support'], df_mining['itemsets'],
                 'o-', color='#E63946', linewidth=3, markersize=10,
                 markerfacecolor='#FFD93D', markeredgecolor='black', markeredgewidth=2)
    axes[1].fill_between(df_mining['min_support'], df_mining['itemsets'],
                         alpha=0.2, color='#E63946')
    axes[1].set_xlabel('Minimum Support Threshold', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Frequent Itemsets', fontsize=12, fontweight='bold')
    axes[1].set_title('Frequent Itemsets vs Min Support', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    for x, y in zip(df_mining['min_support'], df_mining['itemsets']):
        axes[1].annotate(f'{y}', (x, y), textcoords='offset points',
                         xytext=(0, 12), ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUT}/03_mining.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/03_mining.png")

    print("=" * 70)
    print("COMPARISON: APRIORI vs FP-GROWTH")
    print("=" * 70)
    print("""
KEY DIFFERENCES:

FEATURE            APRIORI                  FP-GROWTH
─────────────────────────────────────────────────────────
Approach           Candidate Generation     Divide & Conquer
DB Scans           Multiple (k scans)       Only 2 scans
Data Structure     Hash tree / arrays       FP-Tree
Candidate Sets     YES                      NO
Memory Usage       Lower                    Higher (tree)
Speed              Slower                   Faster
Best For           Small datasets           Large datasets
""")

    # Run both at different support levels
    support_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    comparison = []

    print("\nRunning benchmark...\n")
    for s in support_values:
        # Apriori
        t1 = time.time()
        ap = apriori(df_trans, min_support=s, use_colnames=True, max_len=4)
        ap_time = time.time() - t1

        # FP-Growth
        t2 = time.time()
        fp = fpgrowth(df_trans, min_support=s, use_colnames=True, max_len=4)
        fp_time = time.time() - t2

        speedup = ap_time / fp_time if fp_time > 0 else 0
        comparison.append({
            'min_support': s,
            'Apriori_itemsets': len(ap),
            'FPGrowth_itemsets': len(fp),
            'Apriori_time': round(ap_time, 5),
            'FPGrowth_time': round(fp_time, 5),
            'Speedup': round(speedup, 2)
        })
        print(f"  min_support={s:.3f} | Apriori: {ap_time:.4f}s | FP-Growth: {fp_time:.4f}s | Speedup: {speedup:.2f}x")

    df_comp = pd.DataFrame(comparison)
    print("\n--- Full Comparison Table ---")
    print(df_comp.to_string(index=False))

    # Verify same results
    same_count = all(df_comp['Apriori_itemsets'] == df_comp['FPGrowth_itemsets'])
    print(f"\n✅ Both algorithms produce identical itemsets: {same_count}")

    # ---- VISUALIZATION ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Execution time comparison
    axes[0, 0].plot(df_comp['min_support'], df_comp['Apriori_time'],
                    'o-', color='#E63946', linewidth=3, markersize=10, label='Apriori')
    axes[0, 0].plot(df_comp['min_support'], df_comp['FPGrowth_time'],
                    's-', color='#2E86AB', linewidth=3, markersize=10, label='FP-Growth')
    axes[0, 0].set_xlabel('Minimum Support', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(alpha=0.3)

    # Chart 2: Speedup bar chart
    bars = axes[0, 1].bar(range(len(df_comp)), df_comp['Speedup'],
                          color=plt.cm.Greens(np.linspace(0.4, 0.9, len(df_comp))),
                          edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xticks(range(len(df_comp)))
    axes[0, 1].set_xticklabels([f"{s:.3f}" for s in df_comp['min_support']], rotation=45)
    axes[0, 1].set_xlabel('Minimum Support', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Speedup (Apriori / FP-Growth)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('FP-Growth Speedup Factor', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(df_comp['Speedup']):
        axes[0, 1].text(i, v + 0.05, f'{v:.2f}x', ha='center', fontweight='bold', fontsize=10)

    # Chart 3: Itemsets found (should be equal)
    x = np.arange(len(df_comp))
    width = 0.35
    axes[1, 0].bar(x - width / 2, df_comp['Apriori_itemsets'], width,
                   label='Apriori', color='#E63946', edgecolor='black')
    axes[1, 0].bar(x + width / 2, df_comp['FPGrowth_itemsets'], width,
                   label='FP-Growth', color='#2E86AB', edgecolor='black')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"{s:.3f}" for s in df_comp['min_support']], rotation=45)
    axes[1, 0].set_xlabel('Minimum Support', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Itemsets', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Itemsets Found (Both Should Match)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Chart 4: Time comparison as grouped bars
    axes[1, 1].bar(x - width / 2, df_comp['Apriori_time'], width,
                   label='Apriori', color='#E63946', edgecolor='black')
    axes[1, 1].bar(x + width / 2, df_comp['FPGrowth_time'], width,
                   label='FP-Growth', color='#2E86AB', edgecolor='black')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"{s:.3f}" for s in df_comp['min_support']], rotation=45)
    axes[1, 1].set_xlabel('Minimum Support', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Execution Time (Grouped Bars)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.suptitle('Apriori vs FP-Growth — Complete Comparison',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{OUT}/04_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {OUT}/04_comparison.png")

    # Final summary
    fastest = "FP-Growth" if df_comp['Speedup'].mean() > 1 else "Apriori (this dataset is very small)"
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
📊 BENCHMARK RESULTS:
• Total support thresholds tested: {len(support_values)}
• Both algorithms found identical itemsets: {'✅ YES' if same_count else '❌ NO'}
• Average FP-Growth speedup: {df_comp['Speedup'].mean():.2f}x
• Maximum FP-Growth speedup: {df_comp['Speedup'].max():.2f}x

🏆 FASTEST ON THIS DATASET: {fastest}

FP-Growth in general:
✓ Fewer database scans
✓ No candidate generation
✓ Better for large datasets

⚠️ APRIORI is still useful when:
✓ Dataset is very small
✓ Memory is limited
✓ Simplicity is preferred
""")


try:
    run_fpgrowth_analysis()
except Exception as e:
    print(f"⚠️  The FP-Growth step could not finish: {e}")



print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nFinal Results:")
for name, m in metrics.items():
    print(f"  {name:22s} — Acc: {m['accuracy']:.2%}  F1: {m['f1']:.2%}  AUC: {m['auc']:.3f}")