
# Bank Churn Prediction — Catching Risk Early with Neural Networks

## Summary (For Portfolio)
I built a Neural Network churn model that prioritizes **Recall** so we miss as few churners as possible. Using **SMOTE** to balance the classes and **SGD** optimization, the final model achieved **~0.75 recall on test**, closely matching validation (**~0.7025**), with a low overfitting gap.

### Why it matters
Catching churners early enables proactive retention (offers, outreach). The model’s **low FN (~5%)** means fewer customers slip away unnoticed; FPs are acceptable given lower outreach costs.

### Highlights
- **Best model:** NN + **SMOTE** + **SGD**
- **Test recall ~0.75**, Validation recall ~0.7025
- **Key signals:** Geography (Germany highest churn), inactivity, higher balances among churners, female segment higher churn
- **Action:** Integrate model into CRM to trigger monthly retention workflows

### Tech & Workflow
- Python, scikit‑learn, TensorFlow/Keras, imbalanced‑learn (SMOTE)
- EDA → preprocessing (encoding/normalization) → model baselines → regularization → class balancing → selection by **Recall**
- Reproducible splits (64/16/20) and version‑pinned `requirements.txt`

### Deliverables
- Notebook with full pipeline
- PDF slides summarizing business insights & results
- GitHub repo with README, environment, and reproducible steps
