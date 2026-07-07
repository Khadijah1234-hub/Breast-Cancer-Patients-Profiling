#  Breast Cancer Patient Profiling
## Production-Ready Machine Learning Workflow

![Python](https://img.shields.io/badge/Python-3.8+-teal?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20AI-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-green?style=flat-square)

---

##  Project Overview

Breast cancer diagnosis currently reduces rich quantitative biopsy data into a single binary label — malignant or benign. This loses valuable information about patient sub-groups that may respond differently to treatment protocols.

This project asks a more clinically meaningful question:

> **"Can unsupervised machine learning surface clinically meaningful patient profiles from breast cancer diagnostic data  without using diagnostic labels during learning?"**

Using the **Breast Cancer Wisconsin (Diagnostic) Dataset**, this project builds a complete production-ready ML pipeline that:

1. Discovers hidden patient groups using unsupervised clustering
2. Validates those groups against actual clinical diagnosis
3. Trains supervised classifiers to predict patient profiles for new patients
4. Explains model decisions using SHAP for clinical transparency
5. Deploys the pipeline via FastAPI and Streamlit for real-world use

This is not just a data science exercise — it is a deployable clinical decision support tool designed with African healthcare contexts in mind, where diagnostic resources are often limited and AI can bridge critical gaps.

---

##  Key Results at a Glance

| Metric | Result |
|---|---|
| Natural patient groups discovered | 2 distinct clusters confirmed |
| Unsupervised alignment with actual diagnosis | **91.0% accuracy** |
| Clustering method selected | K-Means (Silhouette: 0.358) |
| Best supervised model (cross-validation) | Logistic Regression (CV: 99.34%) |
| Clinical model accuracy (original labels) | 98% |
| Strongest classifier feature | concavity1 (SHAP importance: 0.233) |
| Optimal classification threshold | 0.3 (Recall: 100%, zero missed malignant) |
| False negatives at optimal threshold | **0** — no malignant patients missed |

---



---

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository (ID: 17) |
| Patients | 569 |
| Features | 30 numeric cell nucleus measurements |
| Target | Diagnosis (M = Malignant, B = Benign) |
| Missing values | None |
| Duplicates | None |
| Class distribution | 357 Benign (62.7%), 212 Malignant (37.3%) |

Each feature is computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

Every measurement is provided in three forms:
- **Mean**  average value across all cell nuclei measured
- **Standard Error (SE)** how consistent the measurements were across nuclei
- **Worst**  the largest or most extreme value recorded

This yields 30 total features per patient.

---

## 🔬 Methodology: Step by Step

### Step 1: Problem Definition

Defined the core research question, objectives, feature descriptions, and success criteria before touching any data.

**Success criteria set upfront:**
- Silhouette Score > 0.25
- Supervised model CV accuracy > 90%
- Clinically interpretable cluster profiles

---

### Step 2: Data Collection & Understanding

Dataset loaded directly from UCI via `ucimlrepo` — no manual file download required.

```python
from ucimlrepo import fetch_ucirepo
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets
```

**Data quality confirmed:**
- 569 patients, 30 features, 0 missing values, 0 duplicates
- All 30 features are float64 — no encoding required
- Diagnosis column (M/B) set aside for post-clustering validation only

---

### Step 3: Train/Test Split — Before Any Preprocessing

**Critical production rule:** The dataset was split before EDA, scaling, or any analysis. This prevents data leakage — the test set must remain completely unseen until final evaluation.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

| Set | Benign | Malignant | Total |
|---|---|---|---|
| Training (80%) | 285 | 170 | 455 |
| Testing (20%) | 72 | 42 | 114 |

Stratified split ensures both sets maintain the same class proportions.

---

### Step 4: Exploratory Data Analysis — Training Set Only

All EDA was performed exclusively on `X_train` to prevent test set patterns from influencing modelling decisions.

**Three visualisations produced:**

**Histograms:**
- Most features show right-skewed distributions — size features have long right tails
- SE features concentrated near zero — measurement error is naturally small
- Bimodal hints visible — early evidence of two patient populations

**Boxplots (scaled):**
- `concavity`, `compactness`, and `concave_points` showed the heaviest outliers
- These outliers are clinically meaningful — representing patients with the most abnormal cell morphology

**Correlation Heatmap:**
- `radius1`, `perimeter1`, `area1` correlated at 0.99 — near-perfect redundancy
- `concavity1` and `concave_points1` = 0.9
- **Conclusion: Severe multicollinearity justifies PCA before clustering**

---

### Step 5: Preprocessing — Training Data Only

StandardScaler fitted on training data only. The learned parameters were then applied to the test set.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_test_scaled = scaler.transform(X_test)         # Transform only
```

All 30 features normalised to mean = 0.0, std = 1.0 on training data.

---

### Step 6: Dimensionality Reduction — PCA

PCA run on all 30 components to determine retention threshold.

| Components | Cumulative Variance |
|---|---|
| PC1 | 44.59% |
| PC1–PC2 | 63.14% |
| PC1–PC5 | 84.73% |
| **PC1–PC10** | **95.21%** |
| PC11–PC30 | < 1% each |

**Decision: 10 components retained** — 95.21% variance, 20 noise components discarded.

- `pca_10` → 10 components for clustering and modelling
- `pca_2` → 2 components for 2D scatter visualisation only

Both fitted on training data and applied to test data without re-fitting.

---

### Step 7: Patient Profiling  K-Means Clustering

Elbow Method confirmed K=2 as optimal; largest inertia drop at this point.

```python
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train_pca)
train_clusters = kmeans.predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)
```

| Set | Cluster 0 | Cluster 1 | Total |
|---|---|---|---|
| Training | 158 | 297 | 455 |
| Testing | 36 | 78 | 114 |

Cluster proportions consistent across training and test — K-Means generalised well.

---

### Step 8: t-SNE Visualisation

t-SNE applied to training and test sets for visualisation only — not included in the prediction pipeline. Both plots confirmed clean cluster separation consistent with PCA findings.

---

### Step 9: Supervised Learning

Three classifiers trained on original 30 scaled features  not PCA components to maintain clinical interpretability.

```python
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}
```

---

### Step 10: Model Evaluation

| Model | Accuracy | ROC-AUC | Misclassifications |
|---|---|---|---|
| Logistic Regression | 99.12% | 1.0000 | 1 |
| Random Forest | 99.12% | 1.0000 | 1 |
| XGBoost | 97.37% | 1.0000 | 3 |

All errors were False Negatives; the most dangerous error type in cancer screening.

---

### Step 11: Cross Validation ( 5-Fold)

| Model | CV Mean | Std | Overfitting |
|---|---|---|---|
| **Logistic Regression** | **99.34%** | **0.0088** | None |
| Random Forest | 97.80% | 0.0197 | Minimal |
| XGBoost | 97.36% | 0.0226 | Minimal |

No significant overfitting detected. **Logistic Regression selected as primary model.**

---

### Step 12: Hyperparameter Tuning (GridSearchCV)

| Model | Best Parameters | CV Score |
|---|---|---|
| Logistic Regression | C=1, solver=lbfgs | 99.34% |
| Random Forest | max_depth=None, n_estimators=50 | 98.02% |
| XGBoost | learning_rate=0.01, max_depth=5 | 97.36% |

Minimal improvement from tuning — default parameters were already near-optimal.

---

### Step 13: Model Explainability (SHAP)

SHAP TreeExplainer applied to tuned Random Forest.

| Feature | SHAP Importance | Clinical Meaning |
|---|---|---|
| concavity1 | 0.233 | High nuclear concavity = malignancy signal |
| concave_points1 | 0.145 | More concave points = malignancy signal |
| concave_points3 | 0.129 | Worst-case concave points = malignancy |
| compactness1 | 0.070 | Dense irregular nuclei = malignancy |
| area3 | 0.036 | Large worst-case area = malignancy |

Nuclear shape irregularity — not size — is the strongest driver of malignant predictions.

---

### Step 14: Threshold Optimisation

| Threshold | Recall | Precision | Accuracy |
|---|---|---|---|
| 0.5 (default) | 97.44% | 100% | 98.25% |
| **0.3 (optimal)** | **100%** | **98.73%** | **99.12%** |

**Threshold 0.3 selected** — zero malignant patients missed.

---

### Step 15: Two-Model Architecture

| Model | Target | Accuracy | Use Case |
|---|---|---|---|
| `lr_cluster_model.pkl` | K-Means clusters | 99% | Patient profiling |
| `lr_clinical_model.pkl` | Actual diagnosis | 98% | Clinical decision support |

---

##  Clinical Interpretation

### Cluster 0: Benign Profile
> Small, compact, uniform tumour cells with low standard error values. Low concavity, low concave points, small area measurements. Strongly associated with **benign tumours**.

### Cluster 1: Malignant Profile
> Large, expansive tumour cells with high concavity, elevated concave points, and dramatically larger worst-case area measurements. High SE values indicate chaotic, inconsistent cell sizes — a hallmark of aggressive cell division. Strongly associated with **malignant tumours**.

### Unsupervised Validation
K-Means, using zero diagnostic labels, independently recovered the benign/malignant distinction with **91% accuracy**.

---

## ⚠️ Limitations

- Dataset is relatively small (569 patients)
- K-Means cluster labels used as supervised target, not true ground truth diagnosis
- Results may not generalise to other populations without external validation
- Human clinical review should always be the final decision layer. this tool supports, not replaces, clinicians

---

##  Future Work

- Validate on larger, more diverse datasets across African healthcare systems
- Explore 3-cluster solution hinted at by dendrogram secondary split
- Integrate genomic and histopathological data for richer patient profiles
- Apply deep learning for more complex pattern detection
- Conduct prospective clinical validation with practicing oncologists
- Deploy as clinical decision support tool for hospitals and health startups in Africa

---

## 🛠️ How To Run

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
jupyter notebook notebooks/Breast_Cancer_ml.ipynb
```

**4. Run all cells top to bottom**

**5. Start the API (optional)**
```bash
uvicorn app:app --reload
```

**6. Open API documentation**
```
http://127.0.0.1:8000/docs
```

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
shap
ucimlrepo
fastapi
uvicorn
streamlit
joblib
jupyter
```

---

##  Author

**Khadijah**
MSc Data and Information Science
Specialisation: AI and Data Science for Health and Life Sciences


---

##  License

This project is open source and available under the MIT License.

---

> *"Data science in service of health equity — surfacing patterns that improve diagnosis, one patient profile at a time."*
