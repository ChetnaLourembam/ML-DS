"""
Kaggle-style linear regression pipeline (Train / Validation / Test)
with (1) lag/rolling features and (2) a fair comparison between:

A) Raw Temperature as a predictor
B) Temperature Anomaly as a predictor

KAGGLE CONCEPT (your setup):
- TRAIN: labels (sales) known → fit model
- VALIDATION: labels known → compare specs and choose best
- TEST: labels unknown (often missing/NaN) → generate predictions for submission
        → DO NOT treat missing test sales as 0

WHY TEMPERATURE CAN "DISAPPEAR" WHEN LAGS ARE ADDED:
- Lag/rolling sales features already encode seasonality and weather effects indirectly.
- Raw temperature is often highly correlated with those lagged/rolling features.
- OLS then attributes most explanatory power to lags, so raw temperature adds little *incremental* value.

TEMPERATURE ANOMALY (recommended for forecasting):
- temp_anomaly = Temperature - average_temperature_for_that_day_of_year
- Removes predictable seasonal temperature patterns
- Preserves "unexpected weather" signal, which is more likely to add value beyond lags

WHAT THIS SCRIPT DOES:
1) Load data
2) Build a modeling table at (Date, Product Group) level:
   - daily_sales target (can be NaN in TEST)
   - calendar/event features (day_of_week, month, holidays, KielerWoche)
   - lag/rolling features per Product Group (leakage-safe via shift)
   - temperature imputed for TEST from historical day-of-year averages
   - temp_anomaly computed from the (imputed) temperature
3) Split into train/val/test by time
4) Compare multiple linear specs on VALIDATION:
   - with raw temperature
   - with temperature anomaly
5) Refit best spec on TRAIN+VALIDATION (common Kaggle practice)
6) Predict TEST and write submission CSV: data_prep/test_predictions_linear_regression.csv

Run from your project folder (where data_prep/ exists):
  & C:/Users/Joalex/anaconda3/python.exe data_regression_kaggle_lr_temp_anomaly.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


# -----------------------------
# 0) Configuration
# -----------------------------
DATA_PATH = Path("data_prep") / "data_org.csv"
OUT_DIR = Path("data_prep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2013-07-01", "2017-07-31"
VAL_START, VAL_END     = "2017-08-01", "2018-07-31"
TEST_START, TEST_END   = "2018-08-01", "2019-07-31"


# -----------------------------
# 1) Metrics
# -----------------------------
def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_mape(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


# -----------------------------
# 2) Build modeling table
# -----------------------------
def build_item_daily_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Return a modeling table with one row per (Date, Product Group).

    Target:
      - daily_sales (float; can be NaN in TEST)

    Features:
      - day_of_week, month
      - is_holiday, KielerWoche
      - Temperature (imputed in TEST using day-of-year historical means)
      - temp_anomaly (Temperature - day-of-year mean)
      - lag/rolling features per Product Group:
          lag_1, lag_7, lag_14
          roll_mean_7, roll_mean_28, roll_std_28
    """
    df = df_raw.copy()

    # Parse types
    df["Date"] = pd.to_datetime(df["Date"])
    df["Sales Volume"] = pd.to_numeric(df.get("Sales Volume"), errors="coerce")

    # Holiday flag from column
    df["is_holiday"] = df["Holiday Name (English)"].notna().astype(int)

    # Temperature may be absent in some rows (especially TEST)
    if "Temperature" in df.columns:
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    else:
        df["Temperature"] = np.nan

    # -------------------------
    # Date-level aggregation
    # -------------------------
    # NOTE: sum(min_count=1) prevents all-NaN from becoming 0.0 (critical for Kaggle TEST).
    df_daily = (
        df.groupby("Date", as_index=False)
          .agg(
              daily_total_sales=("Sales Volume", lambda s: s.sum(min_count=1)),
              is_holiday=("is_holiday", "max"),
              KielerWoche=("KielerWoche", "max"),
              Temperature=("Temperature", "mean"),
          )
    )

    # Calendar features
    df_daily["day_of_week"] = df_daily["Date"].dt.dayofweek
    df_daily["month"] = df_daily["Date"].dt.month
    df_daily["doy"] = df_daily["Date"].dt.dayofyear  # day-of-year for seasonal temperature

    # -------------------------
    # Temperature imputation (for TEST and any missing dates)
    # -------------------------
    # Use historical mean by day-of-year (captures seasonality without needing "future weather").
    doy_mean_temp = (
        df_daily[df_daily["Temperature"].notna()]
        .groupby("doy")["Temperature"]
        .mean()
    )
    df_daily["Temperature"] = df_daily["Temperature"].fillna(df_daily["doy"].map(doy_mean_temp))

    # Fallback if some DOY are still missing (rare): global mean
    df_daily["Temperature"] = df_daily["Temperature"].fillna(df_daily["Temperature"].mean())

    # -------------------------
    # Temperature anomaly (unexpected weather)
    # -------------------------
    # Remove the predictable seasonal pattern so it can add signal beyond lags.
    # Here, doy_mean_temp is a Series indexed by doy; we map it again.
    df_daily["temp_anomaly"] = df_daily["Temperature"] - df_daily["doy"].map(doy_mean_temp)
    df_daily["temp_anomaly"] = df_daily["temp_anomaly"].fillna(0.0)

    # -------------------------
    # Item/day target table
    # -------------------------
    df_item = (
        df.groupby(["Date", "Product Group"], as_index=False)
          .agg(daily_sales=("Sales Volume", lambda s: s.sum(min_count=1)))
    )

    # Merge date-level features into item-level table
    df_item = df_item.merge(
        df_daily[["Date", "Temperature", "temp_anomaly", "day_of_week", "month", "is_holiday", "KielerWoche"]],
        on="Date",
        how="left",
    )

    # -------------------------
    # Lag + rolling features (leakage-safe)
    # -------------------------
    df_item = df_item.sort_values(["Product Group", "Date"]).reset_index(drop=True)
    g = df_item.groupby("Product Group")["daily_sales"]

    df_item["lag_1"] = g.shift(1)
    df_item["lag_7"] = g.shift(7)
    df_item["lag_14"] = g.shift(14)

    df_item["roll_mean_7"] = g.shift(1).rolling(window=7, min_periods=3).mean()
    df_item["roll_mean_28"] = g.shift(1).rolling(window=28, min_periods=7).mean()
    df_item["roll_std_28"] = g.shift(1).rolling(window=28, min_periods=7).std()

    return df_item


# -----------------------------
# 3) Time split
# -----------------------------
def time_split(df_item: pd.DataFrame):
    train = df_item[(df_item["Date"] >= TRAIN_START) & (df_item["Date"] <= TRAIN_END)].copy()
    val   = df_item[(df_item["Date"] >= VAL_START)   & (df_item["Date"] <= VAL_END)].copy()
    test  = df_item[(df_item["Date"] >= TEST_START)  & (df_item["Date"] <= TEST_END)].copy()

    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()

    return train, val, test


# -----------------------------
# 4) Design matrix for OLS
# -----------------------------
def make_X(df_part: pd.DataFrame, feature_cols: list[str], interactions: bool) -> pd.DataFrame:
    """
    Create numeric X for OLS:
    - one-hot encode Product Group
    - fill missing feature values with 0 (important for early test dates with no lag history)
    - optionally add event×product interactions (holidays/KielerWoche affect products differently)
    """
    X = df_part[feature_cols].copy()

    # Fill NaNs in features (lags/rollings may be NaN early in the series)
    X = X.fillna(0)

    if "Product Group" in X.columns:
        X = pd.get_dummies(X, columns=["Product Group"], drop_first=True)

    # bool -> int; cast float
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    X = X.astype(float)

    if interactions:
        event_cols = [c for c in ["is_holiday", "KielerWoche"] if c in X.columns]
        dummy_cols = [c for c in X.columns if c.startswith("Product Group_")]
        for e in event_cols:
            for d in dummy_cols:
                X[f"{e}__x__{d}"] = X[e] * X[d]

    X = sm.add_constant(X, has_constant="add")
    return X


# -----------------------------
# 5) Fit + evaluate specs
# -----------------------------
def fit_eval(train: pd.DataFrame, val: pd.DataFrame, spec_name: str, feature_cols: list[str], interactions: bool):
    """
    Fit on TRAIN (drops rows with missing target),
    Evaluate on VALIDATION (drops rows with missing target).
    """
    tr = train.dropna(subset=["daily_sales"]).copy()
    va = val.dropna(subset=["daily_sales"]).copy()

    # Also ensure features exist; missing are handled by X.fillna(0), but if a column is absent entirely this will error.
    X_train = make_X(tr, feature_cols, interactions=interactions)
    y_train = tr["daily_sales"].astype(float)

    model = sm.OLS(y_train, X_train).fit()

    X_val = make_X(va, feature_cols, interactions=interactions)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)

    y_true = va["daily_sales"].astype(float).to_numpy()
    y_pred = model.predict(X_val).to_numpy()

    return {
        "spec": spec_name,
        "feature_cols": feature_cols,
        "interactions": interactions,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE%": safe_mape(y_true, y_pred),
        "AIC": float(model.aic),
        "BIC": float(model.bic),
        "model": model,
        "train_columns": list(X_train.columns),
    }


def refit_and_predict(best, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Refit best spec on TRAIN+VAL and predict TEST.
    """
    feature_cols = best["feature_cols"]
    interactions = best["interactions"]

    train_full = pd.concat([train, val], axis=0, ignore_index=True)
    train_full = train_full.dropna(subset=["daily_sales"]).copy()

    X_full = make_X(train_full, feature_cols, interactions=interactions)
    y_full = train_full["daily_sales"].astype(float)

    model_full = sm.OLS(y_full, X_full).fit()

    # TEST: keep rows even if target missing (it should be missing); only need features
    te = test.copy()
    X_test = make_X(te, feature_cols, interactions=interactions)
    X_test = X_test.reindex(columns=X_full.columns, fill_value=0.0)

    preds = model_full.predict(X_test).astype(float)
    preds = np.clip(preds, 0.0, None)  # no negative sales

    te = te.copy()
    te["umsatz"] = preds

    # Ensure id exists for submission
    if "id" not in te.columns:
        te["id"] = te["Date"].dt.strftime("%y%m%d") + te["Product Group"].astype(str)

    submission = te[["id", "umsatz"]].sort_values("id")
    return submission


# -----------------------------
# 6) Main
# -----------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run this script from the project root folder (where data_prep/ exists)."
        )

    df_raw = pd.read_csv(DATA_PATH)

    df_item = build_item_daily_table(df_raw)
    train, val, test = time_split(df_item)

    print("\n=== Kaggle split summary ===")
    print(f"TRAIN: {train['Date'].min()} → {train['Date'].max()} | rows={train.shape[0]}")
    print(f"VAL:   {val['Date'].min()} → {val['Date'].max()} | rows={val.shape[0]}")
    print(f"TEST:  {test['Date'].min()} → {test['Date'].max()} | rows={test.shape[0]}")

    # Sanity check: temperature coverage (after imputation)
    print("\n=== Temperature sanity checks ===")
    print("Missing Temperature in TRAIN:", int(train["Temperature"].isna().sum()))
    print("Missing Temperature in TEST:", int(test["Temperature"].isna().sum()))
    print("Missing temp_anomaly in TEST:", int(test["temp_anomaly"].isna().sum()))

    # Candidate specs: direct raw-temp vs anomaly comparison (both with lags/rollings)
    specs = [
        # Baseline lag/rolling without temperature (reference)
        ("lags+rollings + events interactions (no temp)",
         ["Product Group", "day_of_week", "month", "is_holiday", "KielerWoche",
          "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28", "roll_std_28"],
         True),

        # Raw temperature added
        ("lags+rollings + events interactions + RAW temperature",
         ["Product Group", "day_of_week", "month", "is_holiday", "KielerWoche", "Temperature",
          "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28", "roll_std_28"],
         True),

        # Temperature anomaly added (recommended)
        ("lags+rollings + events interactions + TEMP anomaly",
         ["Product Group", "day_of_week", "month", "is_holiday", "KielerWoche", "temp_anomaly",
          "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28", "roll_std_28"],
         True),
    ]

    results = []
    for name, feature_cols, interactions in specs:
        results.append(fit_eval(train, val, name, feature_cols, interactions))

    report = (
        pd.DataFrame([{k: r[k] for k in ["spec","n_train","n_val","MAE","RMSE","MAPE%","AIC","BIC"]} for r in results])
          .sort_values(["MAE","RMSE"])
    )

    print("\n=== Validation comparison (lower is better) ===")
    print(report.to_string(index=False))

    best_spec = report.iloc[0]["spec"]
    best = next(r for r in results if r["spec"] == best_spec)

    print("\n=== Best spec selected (based on validation MAE/RMSE) ===")
    print(best_spec)
    print(best["model"].summary())

    # Refit best spec on train+val and predict test
    submission = refit_and_predict(best, train, val, test)

    out_file = OUT_DIR / "test_predictions_linear_regression.csv"
    submission.to_csv(out_file, index=False)

    print(f"\nSaved submission file to: {out_file}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
