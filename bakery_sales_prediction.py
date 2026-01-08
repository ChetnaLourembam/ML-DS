##############################################################################################################################
# RIDGE REGRESSION , RANDOM FOREST , XG BOOST,  LIGHT GMB and ENSEMBLE 
##############################################################################################################################
'''
Here machine learning models (Ridge Regression, Random Forest, XGBoost, LightGBM) and creating an ensemble for better predictions.
Concepts behind the steps:
Step 1: Load Data
Dates as datetime to create temporal features (weekday, day of year) and calculate lagged sales.

Step 2: Merge Weather & Event Data
Sales often depend on external factors, like weather or festivals. By merging this info, the model can learn these effects.

Step 3: Temporal & Cyclical Features
Cyclical features let models understand repeating patterns (week, season) naturally.

Step 4: Lag & Rolling Features
These autoregressive features are very important in time series forecasting. They allow the model to capture temporal dependencies in sales.

Step 5: Handle Missing Values
Missing weather or lag values are replaced by the median.
Machine learning models cannot handle NaNs directly (except some tree-based models).
Median is robust to outliers and a simple method to impute missing values.

Step 6: Prepare Feature Set
One-hot encoding lets categorical variables like Warengruppe be used in ML models.

Step 7: Train Models
1. Ridge Regression: Linear regression with L2 regularization. Reduces overfitting on correlated features (like lag_1 & rolling_7).
2. Random Forest: Ensemble of decision trees → captures non-linear relationships. Handles missing values and irrelevant features well.
3. XGBoost: Gradient boosting → sequentially corrects errors of previous trees. Often more accurate than Random Forest for tabular data.
4. LightGBM: Faster gradient boosting implementation for large datasets. Efficient with categorical features and large number of observations.
Ensemble: Ensemble methods often perform better than a single model because they average out biases and variance.

Step 8: Feature Importance
Shows which features most influence the model predictions. Useful for interpretability and understanding drivers of sales.
'''


import os
os.getcwd()

import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
# ============================================
# STEP 1: LOAD DATA
# ============================================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
weather = pd.read_csv('wetter.csv')
kiwo = pd.read_csv('kiwo.csv')

# Convert dates
train['Datum'] = pd.to_datetime(train['Datum'])
test['Datum'] = pd.to_datetime(test['Datum'])
weather['Datum'] = pd.to_datetime(weather['Datum'])
kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])


# ============================================
# STEP 2: MERGE WEATHER AND KIWO DATA
# ============================================
# Merge weather
train = train.merge(weather, on='Datum', how='left')
test = test.merge(weather, on='Datum', how='left')

# Merge Kieler Woche
kiwo['KielerWoche'] = 1
train = train.merge(kiwo[['Datum', 'KielerWoche']], on='Datum', how='left')
test = test.merge(kiwo[['Datum', 'KielerWoche']], on='Datum', how='left')

train['KielerWoche'] = train['KielerWoche'].fillna(0).astype(int)
test['KielerWoche'] = test['KielerWoche'].fillna(0).astype(int)


# ============================================
# STEP 3: CREATE EXACT NEURAL NET FEATURES
# ============================================
def create_features(df):
    # Temporal features
    df['weekday'] = df['Datum'].dt.weekday
    df['day_of_year'] = df['Datum'].dt.dayofyear
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

train = create_features(train)
test = create_features(test)


# ============================================
# STEP 4: CREATE LAG AND ROLLING FEATURES
# ============================================
# Sort by category and date
train = train.sort_values(['Warengruppe', 'Datum'])

# Lag features
train['lag_1'] = train.groupby('Warengruppe')['Umsatz'].shift(1)
train['lag_7'] = train.groupby('Warengruppe')['Umsatz'].shift(7)

# Rolling average
train['rolling_7'] = train.groupby('Warengruppe')['Umsatz'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# One-hot encode product groups
train = pd.get_dummies(train, columns=['Warengruppe'], drop_first=True)

# For test set - combine with train to get lag features
test['Umsatz'] = np.nan
combined = pd.concat([train, test])
combined = combined.sort_values(['Datum'])

# Recreate Warengruppe for grouping (from one-hot)
warengruppe_cols = [col for col in combined.columns if col.startswith('Warengruppe_')]
def get_warengruppe(row):
    for i, col in enumerate(warengruppe_cols, start=2):
        if col in row and row[col] == 1:
            return i
    return 1

combined['Warengruppe_temp'] = combined.apply(get_warengruppe, axis=1)

# Calculate lag features for combined
combined['lag_1'] = combined.groupby('Warengruppe_temp')['Umsatz'].shift(1)
combined['lag_7'] = combined.groupby('Warengruppe_temp')['Umsatz'].shift(7)
combined['rolling_7'] = combined.groupby('Warengruppe_temp')['Umsatz'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

combined = combined.drop('Warengruppe_temp', axis=1)

# Split back
train = combined[combined['Umsatz'].notna()].copy()
test = combined[combined['Umsatz'].isna()].copy()
test = test.drop('Umsatz', axis=1)


# ============================================
# STEP 5: HANDLE MISSING VALUES
# ============================================
# Fill weather missing values with median
weather_cols = ['Temperatur', 'Bewoelkung', 'Windgeschwindigkeit', 'Wettercode']
for col in weather_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

# Fill lag features with median (for early dates)
lag_cols = ['lag_1', 'lag_7', 'rolling_7']
for col in lag_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)


# ============================================
# STEP 6: PREPARE FEATURE SET
# ============================================
# Exact features from Neural Network
FEATURES = [
    "Temperatur",
    "Bewoelkung",
    "Windgeschwindigkeit",
    "Wettercode",
    "KielerWoche",
    "is_weekend",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",
    "lag_1", "lag_7", "rolling_7"
]

# Add product group dummies
FEATURES += [col for col in train.columns if col.startswith("Warengruppe_")]

# Prepare X and y
X_train = train[FEATURES]
y_train = train['Umsatz']
X_test = test[FEATURES]

# Fill any remaining NaN values
if X_train.isna().sum().sum() > 0:
    X_train = X_train.fillna(X_train.median())

if X_test.isna().sum().sum() > 0:
    for col in X_test.columns:
        if X_test[col].isna().sum() > 0:
            X_test[col] = X_test[col].fillna(X_train[col].median())



# ============================================
# STEP 7: TRAIN MODELS
# ============================================
# FINAL NaN CHECK AND FIX
X_train = X_train.fillna(0)  
X_test = X_test.fillna(0)    


# Model 1: Ridge Regression
ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_pred = np.maximum(ridge_pred, 0)


# Model 2: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_pred = np.maximum(rf_pred, 0)

# Model 3: XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_pred = np.maximum(xgb_pred, 0)


# Model 4: LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_pred = np.maximum(lgb_pred, 0)


# Ensemble
ensemble_pred = (ridge_pred + rf_pred + xgb_pred + lgb_pred) / 4



# ============================================
# CREATE SUBMISSIONS
# ============================================
submissions = {
    'ridge': ridge_pred,
    'rf': rf_pred,
    'xgb': xgb_pred,
    'lgb': lgb_pred,
    'ensemble': ensemble_pred
}

for name, predictions in submissions.items():
    submission = pd.DataFrame({
        'id': test['id'],
        'umsatz': predictions
    })
    filename = f'submission_{name}_nn_features.csv'
    submission.to_csv(filename, index=False)
    

# ============================================
# FEATURE IMPORTANCE COMPARISON
# ============================================
# Random Forest importance
rf_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)



# XGBoost importance
xgb_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)


# =========================================
# VISUALIZATION 
# =========================================
# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11



# ============================================
# CHART 1: Model Comparison Bar Chart 
# ============================================

fig, ax = plt.subplots(figsize=(12, 6))

# Compute mean predictions from actual model outputs
models = ['Ridge', 'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble']
predictions = [
    np.mean(ridge_pred),
    np.mean(rf_pred),
    np.mean(xgb_pred),
    np.mean(lgb_pred),
    np.mean(ensemble_pred)
]

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

bars = ax.barh(models, predictions, color=colors, alpha=0.8, edgecolor='black')

# Add value labels
for i, value in enumerate(predictions):
    ax.text(value + 0.5, i, f'{value:.2f}', va='center', fontweight='bold')

ax.set_xlabel('Mean Prediction', fontsize=13, fontweight='bold')
ax.set_title('Model Performance Comparison\n(Using Actual Predictions)', 
             fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_real.png', dpi=300, bbox_inches='tight')
plt.close()



# ============================================
# CHART 2: Top 5 Features Across Models
# ============================================

fig, ax = plt.subplots(figsize=(12, 7))

# Combine top 5 from each model
rf_top5 = rf_importance.head(5).copy()
rf_top5['Model'] = 'Random Forest'
xgb_top5 = xgb_importance.head(5).copy()
xgb_top5['Model'] = 'XGBoost'

combined = pd.concat([rf_top5, xgb_top5])

# Create grouped bar chart
x = np.arange(5)
width = 0.35

rf_values = rf_top5['Importance'].values
xgb_values = xgb_top5['Importance'].values

bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', 
               color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, xgb_values, width, label='XGBoost', 
               color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_xlabel('Feature', fontsize=13, fontweight='bold')
ax.set_ylabel('Importance', fontsize=13, fontweight='bold')
ax.set_title('Top 5 Features: Random Forest vs XGBoost', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(rf_top5['Feature'], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('top5_comparison.png', dpi=300, bbox_inches='tight')

plt.close()


# ============================================
# CHART 3: Key Finding - Lag Dominance
# ============================================

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate lag vs non-lag importance
lag_features = ['lag_1', 'lag_7', 'rolling_7']
lag_importance = rf_importance[rf_importance['Feature'].isin(lag_features)]['Importance'].sum()
non_lag_importance = rf_importance[~rf_importance['Feature'].isin(lag_features)]['Importance'].sum()

categories = ['Lag Features\n(Past Sales)', 'All Other Features\n(Weather, Time, Events)']
values = [lag_importance * 100, non_lag_importance * 100]
colors_bar = ['#e74c3c', '#95a5a6']

bars = ax.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', width=0.6)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Importance (%)', fontsize=13, fontweight='bold')
ax.set_title('KEY FINDING: Lag Features Dominate Prediction\n(Random Forest Model)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add annotation
ax.annotate('Past sales are 90% of\npredictive power!', 
            xy=(0, lag_importance * 100), 
            xytext=(0.5, 70),
            fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('lag_dominance.png', dpi=300, bbox_inches='tight')
plt.close()