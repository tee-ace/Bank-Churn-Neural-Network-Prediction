# Bank-Churn-Neural-Network-Prediction[README_churn.md](https://github.com/user-attachments/files/22032840/README_churn.md)

# Bank Churn Prediction — Neural Networks (SMOTE + SGD)

## Project Overview
Predict customer churn (exit within 6 months) for a retail bank and surface the drivers of churn so the business can run targeted retention plays.

- **Business goal:** Reduce missed churners (false negatives) so the bank can intervene early.
- **Primary metric:** **Recall** on the churn class.
- **Final model:** Neural Network with **SMOTE** (to balance classes) + **SGD** optimizer.
- **Why this model:** Best validation recall (≈ **0.7025**) with the **lowest overfitting gap**; test recall ≈ **0.75**.

## Key Results
- **Test recall ≈ 0.75**, validation recall ≈ **0.7025**.
- **False Negatives ~5% (≈102)** — aligns with the business goal to catch churners early.
- **False Positives ~23% (≈459)** — acceptable tradeoff given the lower cost of outreach vs. losing a customer.
- **305** actual churners were correctly identified in test predictions.

## Actionable Insights (Selected)
- **Geography matters:** Germany churn rate ≈ **32.44%**; France lowest at ≈ **16.15%** (despite >50% of base).
- **Inactive members churn more:** ~**26.85%** vs **14.26%** for active.
- **Higher balances among churners** (counter‑intuitive; warrants deeper product/experience review).
- **Females churn more than males** (≈ **25.07%** vs **16.45%**).

## Data
- **Rows:** ~10,000 customers; **Target:** `Exited` (1 = churned).
- **Notable fields:** `Geography`, `Gender`, `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
- **Imbalance:** ~**20.4%** churners.

## Method
1. **EDA** — class imbalance, country/gender/engagement patterns; no missing values.
2. **Preprocessing**
   - Drop identifiers; one‑hot encode categoricals (`Geography`, `Gender`).
   - Normalize numeric features.
   - Train/val/test split: **64/16/20** (random_state=42).
3. **Models**
   - Baselines (NN + SGD/Adam), regularization (dropout), **SMOTE** for balancing.
   - Tracked **Recall** as the primary metric.
4. **Model selection**
   - Picked model with highest **validation recall** and smallest train–val gap.
   - **Chosen:** NN + **SMOTE** + **SGD**.

## Final NN (concise)
- Input dims: 11 engineered features
- Hidden: [32, 16, 8] ReLU (variants tried)
- Output: Sigmoid
- Optimizer: **SGD(lr=1e-3)**; Loss: BCE; Epochs: 100; Batch: 32
- Class balancing: **SMOTE** on training split

## How to Run
```bash
git clone https://github.com/yourusername/bank-churn-nn.git
cd bank-churn-nn
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
# open notebooks/Bank_Churn_Project_Notebook.ipynb
```

## Repo Structure
```
bank-churn-nn/
├─ data/                 # (optional placeholder; add .gitkeep, not raw PII)
├─ notebooks/
│  └─ Bank_Churn_Project_Notebook.ipynb
├─ reports/
│  └─ Bank_Churn_Project_Presentation.pdf
├─ src/                  # (optional: model training/eval scripts)
├─ README.md
├─ requirements.txt
└─ LICENSE
```

## Requirements
See `requirements.txt` for exact versions.

## Notes
- **Why Recall?** Missing a churner (FN) is costlier than a false alarm (FP). Maximizing recall reduces FN.
- **Tradeoff:** Higher recall increased FPs (~23%), but this is a cheaper cost (targeted outreach) than lost revenue.
