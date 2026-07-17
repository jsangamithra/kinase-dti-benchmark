# kinase-dti-benchmark
Benchmarking a dual-branch neural network against classical ML (Random Forest, XGBoost, SVM) for kinase inhibitor drug-target interaction prediction — with cold-start generalization testing, SHAP explainability, and real-world clinical validation on Davis &amp; KIBA.
# Drug-Target Interaction (DTI) Prediction for Kinase Inhibitors

A benchmarking study comparing a custom **Dual-Branch Neural Network** against classical machine learning baselines (**Random Forest**, **XGBoost**, **Bagged SVM**) for predicting drug-target interactions, evaluated on the **Davis** and **KIBA** kinase inhibitor datasets under warm-split and cold-start (unseen-drug / unseen-target) generalization conditions.

> M.Sc. Bioinformatics research project — feature-based DTI prediction with explainability (SHAP) and real-world clinical validation.

---

## Overview

Kinases are among the most heavily targeted protein families in cancer drug discovery. This project builds a complete, end-to-end pipeline for predicting whether a candidate drug molecule will interact with a given kinase target, and — more importantly — rigorously tests how well that prediction generalizes to **drugs and targets the model has never seen before**, which is the scenario that actually matters in real drug discovery.

**Key questions this project answers:**
- How well do classical ML models vs. a dual-branch deep learning model perform on standard warm-split benchmarks?
- How much does performance degrade when tested on entirely novel drugs or novel protein targets?
- What features (drug chemistry vs. protein sequence) does each model actually rely on, and does that explain its generalization behavior?
- Do the trained models correctly predict real, clinically approved kinase inhibitors they've never seen?

---

## Key Results

| Model | Davis AUC (warm) | Davis AUC (cold-drug) | KIBA AUC (warm) | KIBA AUC (cold-drug) |
|---|---|---|---|---|
| Random Forest | 0.889 | 0.758 | 0.979 | — |
| XGBoost | 0.927 | 0.743 | 0.996 | — |
| Bagged SVM | 0.932 | 0.788 | 0.993 | — |
| **Dual-Branch NN** | **0.932** | **0.725** | **0.995** | **0.936** |

- The Dual-Branch Neural Network shows a **statistically significant improvement over Random Forest** on both datasets (5-fold CV, p < 0.01).
- **Generalizing to unseen drugs is substantially harder than generalizing to unseen targets** — especially on Davis, where AUC drops from 0.93 (warm) to 0.72 (cold-drug), versus only a 0.05 drop on cold-target.
- SHAP analysis shows the Dual-Branch NN relies mainly on **drug fingerprint features**, while Random Forest relies mainly on **protein embedding features** — directly explaining why each model degrades differently under cold-start conditions.
- In a real-world case study, the Davis-trained model correctly identified **3 of 3 novel, FDA-approved kinase inhibitors** (Asciminib, Osimertinib, Lorlatinib) it had never seen during training.

Full results, figures, and discussion are available in the project report / thesis (see [`docs/`](./docs)).

---

## Datasets

| Dataset | Pairs | Unique Drugs | Unique Targets | Binarization |
|---|---|---|---|---|
| [Davis](https://www.nature.com/articles/nbt.1990) | 25,862 | 68 | 381 (332 unique sequences) | pKd ≥ 7.0 |
| [KIBA](https://pubs.acs.org/doi/10.1021/ci400709d) | 14,547 | 1,679 | 205 | KIBA score ≤ 12.1 |

> **Note:** During data auditing, a label-polarity inconsistency was identified in the Davis binarization step (see [Known Issues](#known-issues) below). KIBA was verified to be labeled correctly.

---

## Methodology

**Drug representation:** Morgan (ECFP4) circular fingerprints, generated via RDKit from canonical SMILES.

**Protein representation:** [ESM-2](https://github.com/facebookresearch/esm) protein language model embeddings (`esm2_t6_8M_UR50D`, 320-dimensional).

**Models:**
- Random Forest (300 trees, max depth 12, balanced class weights)
- XGBoost (500 rounds, max depth 6, lr 0.05, `scale_pos_weight` tuned per dataset)
- Bagged SVM (10 × RBF-kernel SVC, bootstrap-sampled to 5,000 rows each, for scalability)
- **Dual-Branch Neural Network** — separate drug branch (→1024→512) and protein branch (→256→128), fused (640→256→64→1), ~1.7–1.9M parameters

**Evaluation splits:**
- **Warm split** — random split (best-case, optimistic)
- **Cold-drug split** — test-set drugs excluded from training (tests generalization to novel chemistry)
- **Cold-target split** — test-set targets excluded from training (tests generalization to novel proteins)

**Additional analysis:**
- 5-fold cross-validation with paired statistical significance testing
- SHAP explainability, including cross-dataset feature-importance stability analysis
- Real-world case study against 5 known drug-target pairs, including 3 clinically approved kinase inhibitors never seen in training

---

## Repository Structure

```
├── notebooks/
│   ├── 01_data_loading_eda.ipynb          # Data loading & exploratory analysis
│   ├── 02_preprocessing_splits.ipynb      # Warm / cold-drug / cold-target split generation
│   ├── 03_feature_extraction.ipynb        # Morgan fingerprints + ESM-2 embeddings
│   ├── 04_model_training_cv.ipynb         # Model training + 5-fold cross-validation
│   ├── 05_shap_explainability.ipynb       # SHAP analysis
│   └── 06_case_study.ipynb                # Clinical drug-target case study
├── data/
│   ├── davis/                             # Raw + processed Davis dataset
│   └── kiba/                              # Raw + processed KIBA dataset
├── models/
│   ├── davis/                             # Trained model artifacts (.pkl / .pth)
│   └── kiba/
├── results/
│   ├── davis/ , kiba/ , final/            # Metrics, SHAP values, CV results (.csv / .json)
├── figures/
│   ├── davis/ , kiba/ , shared/           # All generated plots
├── docs/
│   └── report.docx                        # Full project report / thesis
└── README.md
```

*(Adjust folder names above if your actual repo layout differs.)*

---

## Installation

```bash
git clone https://github.com/jsangamithra/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Core dependencies:** `rdkit`, `fair-esm`, `torch`, `xgboost`, `scikit-learn`, `shap`, `pandas`, `numpy`, `matplotlib`

---

## Usage

Run the notebooks in order (01 → 06), or load a pretrained model directly:

```python
import joblib

model = joblib.load("models/davis/XGBoost_warm.pkl")
predictions = model.predict_proba(X)[:, 1]
```

For the Dual-Branch Neural Network:

```python
import torch
from model_def import DualBranchNN  # architecture defined in notebooks/04

model = DualBranchNN(drug_dim=1043, prot_dim=320)
model.load_state_dict(torch.load("models/davis/DualBranch_davis_warm.pth"))
model.eval()
```

---

## Known Issues

- **Davis label polarity:** A cross-check of raw Kd-derived pKd values against assigned binary labels suggests the "interacting" (`1`) and "non-interacting" (`0`) labels may be inverted relative to the standard pKd ≥ 7.0 convention for the Davis dataset specifically. KIBA was independently verified to be labeled correctly. This does not affect AUC/ranking-based metrics but affects the biological interpretation of "positive class." **Verification/fix pending** — see `notebooks/02_preprocessing_splits.ipynb`.
- **Kinase point-mutant sequences:** Named ABL1 point-mutant variants (e.g., `ABL1(T315I)`, `ABL1(E255K)`) currently share an identical underlying protein sequence in the Davis data, so the protein branch cannot distinguish mutation-specific binding effects. Mutation-aware sequence encoding is planned as future work.

---

## Future Work

- Fix and re-validate the Davis label-polarity issue
- Mutation-aware protein sequence encoding for kinase resistance variants
- Replace Morgan fingerprints with pretrained molecular embeddings (ChemBERTa / MolCLR) to improve cold-drug generalization
- Expand the clinical case study to a larger curated set of drug-target pairs

---

## Author

**Sangamithra J**
M.Sc. Bioinformatics, Stella Maris College, Chennai
[GitHub](https://github.com/jsangamithra)

---

## License

*(Add your preferred license, e.g., MIT — see [choosealicense.com](https://choosealicense.com/) if unsure.)*
