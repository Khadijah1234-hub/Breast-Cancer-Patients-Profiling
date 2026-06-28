# 🔬 Breast Cancer Patient Profiling
## Using Principal Component Analysis (PCA) and Unsupervised Clustering

![Python](https://img.shields.io/badge/Python-3.8+-teal?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-blue?style=flat-square)
![Type](https://img.shields.io/badge/Learning-Unsupervised%20%2B%20Supervised-coral?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)

---

## Project Overview

Breast cancer remains one of the most commonly diagnosed cancers worldwide. Standard diagnostic workflows reduce rich quantitative biopsy data into a single binary label,  malignant or benign. This project asks a more nuanced question:

> **"Beyond the binary label, what hidden patient sub-groups exist within breast cancer diagnostic data, and can unsupervised machine learning surface clinically meaningful patterns that supervised classification alone would miss?"**

Using the **Breast Cancer Wisconsin (Diagnostic) Dataset**, this project applies dimensionality reduction and clustering algorithms to discover natural patient groupings  without using diagnostic labels during learning. A supervised classification layer is then built on top of the discovered clusters to create an interpretable, deployable patient profiling model.

---

##  Key Results

| Finding | Result |
|---|---|
| Natural patient groups discovered | 2 distinct clusters confirmed |
| Unsupervised alignment with diagnosis | **91.0% accuracy** |
| Top clustering method | K-Means (Silhouette: 0.358) |
| Top supervised model | Logistic Regression (CV: 98.95%) |
| Strongest classifier feature | concavity1 (importance: 0.189) |
| Strongest separator feature | area3 (mean difference: 854.47) |
| Outlier patients identified | 33 borderline cases (DBSCAN) |

---

##  Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**
- **Source:** UCI Machine Learning Repository (ID: 17)
- **Patients:** 569
- **Features:** 30 numeric cell nucleus measurements
- **Target:** Diagnosis (M = Malignant, B = Benign); used only for validation
- **Missing values:** None

Each feature is computed from a digitised image of a fine needle aspirate (FNA) of a breast mass and describes characteristics of the cell nuclei present  including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each measurement is provided as mean, standard error, and worst value yielding 30 total features.

---

## 🔬 Methodology

### Step 1: Exploratory Data Analysis (EDA)
- Histograms revealed right-skewed distributions in size features (area, perimeter, radius)
- Boxplots confirmed heavy outliers in irregularity features (concavity, compactness, concave points)
- Correlation heatmap revealed severe multicollinearity — radius, perimeter, and area correlated at 0.99
- Finding: Multicollinearity justified PCA before clustering

### Step 2: Preprocessing
- StandardScaler applied to all 30 features
- Mean = 0.0, Std = 1.0 confirmed for all features
- Diagnosis column set aside for post-clustering validation only

### Step 3: Dimensionality Reduction (PCA)
- PCA run on all 30 components to assess explained variance
- **10 components retain 95.16% of total variance**
- 2 components used for visualisation (63.24% variance)
- Components 11-30 discarded as noise (each < 1%)

| Components | Cumulative Variance |
|---|---|
| PC1 | 44.27% |
| PC1–PC2 | 63.24% |
| PC1–PC5 | 84.73% |
| PC1–PC10 | 95.16% |

### Step 4: Visualisation
- PCA scatter plot revealed two natural groups separated around PC1 = 0
- t-SNE confirmed the same two groups with cleaner separation

### Step 5: Clustering

**K-Means**
- Elbow Method tested K = 1 to 10
- Largest inertia drop at K = 2, selected as optimal
- Result: Cluster 0 = 380 patients, Cluster 1 = 189 patients

**Hierarchical Clustering**
- Dendrogram showed biggest split at distance ~99, confirmed K = 2
- Result: Cluster 0 = 244 patients, Cluster 1 = 325 patients
- Both methods independently agreed on two natural groups


### Step 6: Cluster Evaluation

| Method | Silhouette Score | Davies-Bouldin Score |
|---|---|---|
| **K-Means** | **0.3580** | **1.2597** |
| Hierarchical | 0.2960 | 1.3805 |

**K-Means selected** as primary clustering method, superior on both metrics.

### Step 7: Feature Importance Analysis

**Unsupervised (mean difference between clusters):**

| Rank | Feature | Difference |
|---|---|---|
| 1 | area3 | 854.47 |
| 2 | area1 | 507.28 |
| 3 | area2 | 54.93 |

**Supervised (Random Forest importance):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | concavity1 | 0.189 |
| 2 | concave_points1 | 0.158 |
| 3 | concave_points3 | 0.139 |

> Size features dominate unsupervised analysis due to raw scale.
> Irregularity features dominate supervised analysis due to decision boundary power.
> Both findings are valid,  they answer different clinical questions.

### Step 8: Validation

| Cluster | Benign | Malignant | Accuracy |
|---|---|---|---|
| Cluster 0 (380 patients) | 343 | 37 | 90.3% |
| Cluster 1 (189 patients) | 14 | 175 | 92.6% |
| **Overall** | **357** | **212** | **91.0%** |

K-Means achieved 91% alignment with actual diagnosis without ever seeing the labels.

### Step 9: Supervised Classification

| Model | Accuracy | ROC-AUC | CV Mean | CV Std |
|---|---|---|---|---|
| Logistic Regression | 100% | 1.0000 | 98.95% | 0.0066 |
| Random Forest | 100% | 1.0000 | 97.54% | 0.0087 |
| XGBoost | 97% | 0.9989 | 97.01% | 0.0199 |

**Final selected model: Logistic Regression**  most stable across cross validation folds, confirming cluster boundaries are largely linear.

---

##  Clinical Interpretation

### Cluster 0: 380 Patients (66.8%)
> Patients present with small, compact, uniform tumour cells with low standard error values indicating highly consistent cell measurements across nuclei. This profile is strongly associated with **benign tumours**.

### Cluster 1: 189 Patients (33.2%)
> Patients present with large, expansive tumour cells with dramatically higher worst-case measurements. The elevated area3 values suggest nuclei that have grown far beyond normal size — a hallmark of uncontrolled, aggressive cell division. This profile is strongly associated with **malignant tumours**.

### Key Clinical Findings

| Finding | Clinical Meaning |
|---|---|
| 91% unsupervised accuracy | ML can profile patients without diagnostic labels |
| concavity1 top RF feature | Nuclear shape irregularity is key malignancy marker |
| area3 top unsupervised feature | Worst-case tumour size signals malignancy |
| Two confirmed clusters | Data supports binary malignant/benign profiling |

---

## ⚠️ Limitations

- Dataset is relatively small (569 patients)
- K-Means cluster labels used as supervised target — not true ground truth diagnosis
- Perfect train/test scores suggest possible overfitting to cluster boundaries, cross validation confirmed mild overfitting only
- Results may not generalise to other breast cancer datasets or populations without further validation

---

##  Future Work

- Validate findings on larger, more diverse patient datasets
- Apply deep learning for more complex pattern detection
- Extend analysis to genomic and histopathological data
- Deploy as a clinical decision support tool for health facilities in resource-limited settings across Africa

---

##  How To Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/Breast-Cancer-Patient-Profiling.git
cd Breast-Cancer-Patient-Profiling
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook notebooks/breast_cancer_profiling.ipynb
```

**4. Run all cells top to bottom**

---

##  Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
ucimlrepo
jupyter
```

---

## 👩🏾‍💻 Author

**Khadijah**
MSc Data and Information Science
Specialisation: AI and Data Science for Health and Life Sciences in Africa

---

## 📄 License

This project is open source and available under the MIT License.

---

> *"Data science in service of health equity, surfacing patterns that improve diagnosis, one patient profile at a time."*
