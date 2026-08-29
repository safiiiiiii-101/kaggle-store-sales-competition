# Store Sales — Time Series Forecasting

Kaggle "Getting Started" competition — forecasting daily unit sales for thousands of product families across 54 stores for Corporación Favorita, an Ecuadorian grocery retailer.

**Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

## Problem

Predict `sales` (unit sales per store, per product family, per day) for a 16-day window immediately following the training period. ~3 million training rows (54 stores × 33 product families × ~1,684 days). Evaluated on RMSLE (Root Mean Squared Logarithmic Error), so relative error matters more than absolute error.

## Data

Seven files, not one flat table — a realistic relational setup that had to be joined manually:
- `train.csv` / `test.csv` — the core sales data
- `stores.csv` — store metadata (city, state, type, cluster)
- `oil.csv` — daily oil price (Ecuador's economy is oil-dependent, used as a macro signal)
- `holidays_events.csv` — national/regional/local holidays and calendar events
- `transactions.csv` — daily transaction counts (train period only, not usable as a model feature since it doesn't exist for test dates)

## Why preprocessing was the hard part of this competition

This competition is deceptively difficult for a "getting started" comp — almost all of the real difficulty is in correctly preparing the data, not in the modeling itself. A few specific things that took real debugging:

**Oil price gaps.** `oil.csv` only has ~1,218 rows for a period that spans 1,704 calendar days — it excludes weekends and some market holidays entirely, since oil markets don't trade then. Grocery sales happen every day, so the missing rows had to be reconstructed first (building a full daily date range and merging real prices onto it) before forward/backward-filling could correctly close the remaining gaps. Filling gaps in the raw file alone wasn't enough — the missing *rows*, not just missing *values*, were the real problem.

**Holiday logic.** Not every row in `holidays_events.csv` represents an actual day off. Some holidays are marked `transferred = True`, meaning the observance moved to a different date — the original date was a normal working day, and a separate `Transfer` row elsewhere in the file is the real day off. Filtering this correctly (and handling `Bridge`/`Additional`/`Work Day` types) required real logic, not just reading the `type` column at face value. National, regional, and local holidays also needed separate merge keys (`date` alone, `date`+`state`, `date`+`city` respectively), since each applies to a different scope of stores.

**Ordering bugs from encoding too early.** Categorical columns (`city`, `state`, `family`, `type`) were label-encoded into numbers for the model — but this had to happen *after* merging holiday data, since the holiday file's location names are text (`"Manta"`, `"Imbabura"`) and won't match against already-encoded integer codes. Encoding too early silently broke the holiday merge with a type-mismatch error that took real tracing to catch and fix.

**Leakage risk in engineered features.** Rolling-mean features are especially easy to leak by accident — computing a rolling average without first shifting the window by one day means "today's" feature would include today's own sales value, handing the model part of the answer it's supposed to predict. Verified this was handled correctly by manually tracing the math on a single store/product combination before trusting it at scale.

## Approach

- Reconstructed a complete daily date range for oil prices, merged real prices onto it, then forward/backward-filled remaining gaps
- Filtered holidays to only genuinely-observed days (handling `transferred`/`Transfer`/`Bridge` logic), split into national/regional/local lookup tables, each merged with the correct key
- Engineered date features (year, month, day, day of week)
- Added lag features (`lag_7`, `lag_14`, `lag_21`, `lag_28`) and a rolling 7-day mean, computed per store+product family, with train and test concatenated first so test's early days could correctly reference train's most recent history
- Verified no leakage in the rolling/lag features by manually checking the math against a single store/family group
- Trained an XGBoost regressor on `log1p(sales)` (log-transforming the target to align the model's loss function with RMSLE), evaluated with a time-based (not random) train/validation split, since a random split would let the model train on rows chronologically after its own validation set

## Results

| Version | Validation RMSLE |
|---|---|
| Baseline (no holidays, basic oil fill) | ~0.56 |
| + holiday flags | ~0.46 (original full pipeline) |
| + lag & rolling-mean features | **0.386** |

Lag and rolling-mean features gave by far the largest single improvement — a much bigger jump than holiday or oil-price handling — confirming that recent sales history for a given store+product is the strongest available predictor of near-term future sales.

## Files

- `store-sales-forecasting.ipynb` — full notebook (data cleaning, merges, feature engineering, model training, validation)

## Tools

Python, pandas, NumPy, XGBoost, scikit-learn (`mean_squared_log_error`)
