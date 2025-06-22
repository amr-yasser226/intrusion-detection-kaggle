# Cyber-Attack Classification 

This repository contains the end-to-end pipeline, analysis, and results for the **CSAI 253 – Machine Learning** course project (Phase 2). We built and evaluated multiple tree‐based and ensemble classifiers to distinguish between benign and various attack types (DDoS, DoS, Mirai, Recon, MITM) using per-flow network features. The project was organized into data preparation, exploratory analysis, feature engineering, imbalance handling, outlier treatment, modeling, and final submission to the Kaggle competition [csai-253-project-phase-2](https://www.kaggle.com/competitions/csai-253-project-phase-2).

---

## Repository Structure

```

.
├── cache/                         # Temporary files and CatBoost logs
├── catboost\_info/                 # TensorBoard event logs for CatBoost training
├── data/
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Test data
│   ├── phase2\_students\_before\_cleaning.csv
│   ├── sample\_submission.csv
│   └── Our\_Competition\_Submission.csv  # Final Kaggle submission (Private 0.9163 / Public 0.9146)
├── figures/
│   ├── class\_distribution.png
│   ├── correlation\_matrix.png
│   └── feature\_importance.png
├── imbalance\_analysis/            # Imbalance diagnostics and plots
├── Models/                        # Model artifacts and notebooks
│   ├── scaler.joblib
│   ├── selector.joblib
│   ├── xgb\_model.joblib
│   ├── stacking\_model.joblib
│   └── \*.ipynb
├── notebooks/                     # Data profiling, cleaning, and preprocessing notebooks
│   ├── data\_profilling.ipynb
│   ├── Feature\_Descriptions.ipynb
│   ├── handling\_duplicates.ipynb
│   ├── handling\_imbalance.ipynb
│   ├── handling\_outliers.ipynb
│   ├── model.ipynb
│   ├── scaling.ipynb
│   └── ydata\_profiling\_code.ipynb
├── Report/                        # PDF reports on methodology and rationale
│   ├── Columns Report.pdf
│   ├── Encoding Techniques.pdf
│   ├── Feature Descriptions & Preprocessing Report.pdf
│   ├── Feature Engineering Report.pdf
│   ├── \[FINAL] PHASE 2 REPORT.pdf
│   ├── Handling Duplicates.pdf
│   ├── Handling Outliers.pdf
│   ├── Models Scaling.pdf
│   ├── Numerical Features Skewness Report.pdf
│   ├── Proper Treatment of Test Data in SMOTE Workflows.pdf
│   ├── Why You Should Split Your Data Before Correcting Skewness.pdf
│   └── skewness\_report.txt
├── LICENSE
└── README.md

````

---

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/amr-yasser226/intrusion-detection-kaggle.git
   cd Machine-Learning-Phase2

2. **Dependencies**
   A typical environment includes:

   * Python 3.8+
   * `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`, `ydata-profiling`, `optuna`, `matplotlib`, `seaborn`, `joblib`, `pdfkit`

3. **Data**

   * Place `train.csv` and `test.csv` in `/data`.
   * Inspect `phase2_students_before_cleaning.csv` for raw, uncleaned data.

4. **Exploratory Analysis & Profiling**

   * Run `notebooks/data_profilling.ipynb` to generate profiling reports.
   * Visualize distributions, skewness, and correlations.

5. **Preprocessing Pipelines**

   * **Deduplication**: `handling_duplicates.ipynb` explores direct removal, weighting, and train–test aware grouping.
   * **Skew correction**: log1p, Yeo–Johnson, Box–Cox — always fit on training split only (`Why You Should Split Your Data…`).
   * **Outlier treatment**: winsorization, Z-score, isolation forest (`handling_outliers.ipynb`).
   * **Scaling**: Standard, MinMax, Robust, Quantile (`scaling.ipynb`).
   * **Imbalance handling**: SMOTE, SMOTE-Tomek, ClassWeights, EasyEnsemble, RUSBoost (`handling_imbalance.ipynb`).

6. **Feature Engineering**

   * Additional features (e.g. `rate_ratio`, `avg_pkt_size`, `burstiness`, `payload_entropy`, time-cyclic features) in `Feature Engineering Report.pdf` and implemented in `scaling.ipynb`.

7. **Model Training & Evaluation**

   * **XGBoost** and **Stacking** in `Model.ipynb` / `Phase_2 model.ipynb`.
   * Hyperparameter tuning and Optuna‐based lightGBM/CatBoost ensembles in `data_profilling.ipynb`.
   * Final models saved as `.joblib` in `/Models/`.

8. **Results & Submission**

   * Final private score: **0.916289**, public score: **0.914581** on Kaggle.
   * Submission file: `data/Our_Competition_Submission.csv`.

---

## Key Findings

* **Deduplication** first prevents leakage and skewed statistics.
* **Skew correction** must be fit only on the training data to avoid over-optimistic metrics.
* **Tree-based models** are largely scale-invariant, but scaling benefits pipelines that mix learners.
* **Outlier handling** (winsorization, isolation forest) improves model robustness.
* **Class imbalance** addressed via SMOTE (training only) and ensemble methods.
* **XGBoost** with tuned hyperparameters achieved the best standalone performance; stacking did not outperform it.

---

## How to Reproduce

1. **Run the notebooks in order** within a Jupyter environment, starting with data profiling and ending with `model.ipynb`.
2. **Generate figures** in `/figures` and `/imbalance_analysis`.
3. **Train final models** and export `xgb_model.joblib`, `stacking_model.joblib`.
4. **Create submission** by loading `test.csv`, applying preprocessing, predicting, and saving `Our_Competition_Submission.csv`.

---

## License

This project is released under the [MIT License](LICENSE).

