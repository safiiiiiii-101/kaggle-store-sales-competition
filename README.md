# Bank Term Deposit Subscription — Binary Classification

Kaggle Playground Series (S5E8) — predicting whether a bank client will subscribe to a term deposit, based on marketing campaign data.

**Competition:** [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)

## Problem

750,000 training rows, 16 features (demographics, account info, campaign contact history), predicting a binary target `y` (subscribed or not). Evaluated on AUC.

The dataset is imbalanced — only ~12% of clients subscribed, ~88% did not.

## Approach

- Cleaned and encoded 9 categorical columns (job, marital, education, default, housing, loan, contact, month, poutcome) using label encoding, verifying train/test category mappings matched
- Trained an XGBoost classifier (`binary:logistic`, AUC eval metric)
- Used a stratified train/validation split to preserve the true class ratio, since a plain random split risks uneven class distribution given the imbalance
- Validated results with 5-fold stratified cross-validation to confirm the model's performance was stable, not a lucky/unlucky single split

## Key finding: `duration` and data leakage

The `duration` column (length of the last contact call, in seconds) turned out to be the single most predictive feature — average call duration was **638s for subscribers vs 204s for non-subscribers**, a ~3x gap.

The problem: call duration is only known *after* the call happens. A model using it can't realistically be used to decide *who to call* in the first place — the actual business use case. This isn't leakage in the sense of a data pipeline bug (the column is legitimate, correctly recorded data), but it is a temporal leakage problem relative to the real prediction task.

I trained and compared both versions:

| Model | AUC (validation) | AUC (Kaggle public LB) |
|---|---|---|
| With `duration` | 0.966 | 0.96727 |
| Without `duration` | 0.850 (5-fold mean: 0.853) | 0.85191 |

Removing `duration` drops AUC by ~0.116 — quantifying exactly how much of the "with duration" score was inflated by a feature that isn't realistically available at prediction time.

Without `duration`, the model's top features become `contact` method, previous campaign outcome (`poutcome`), and `housing` loan status — all genuinely known before contacting a client, and each with an intuitive real-world explanation (e.g. clients without a housing loan subscribe at roughly double the rate of those with one).

## Cross-validation results (no duration model)

5-fold stratified CV, mean AUC 0.853, folds ranging 0.8509–0.8566 — tight spread, indicating a stable estimate rather than one lucky split.

## Files

- `bank-classification.ipynb` — full notebook (EDA, preprocessing, both model versions, feature importance, CV)

## Tools

Python, pandas, XGBoost, scikit-learn (StratifiedKFold, train_test_split, roc_auc_score)


