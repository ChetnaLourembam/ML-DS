#######################################################################
# Machine Learning- Backery Sales Prediction #
#######################################################################
import os
os.getcwd()

import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Dates
from datetime import datetime, timedelta

# Machine Learning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Boosting 
import xgboost as xgb
import lightgbm as lgb

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)

class BakerySalesForecaster:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.weather_df = None
        self.kiwo_df = None
        self.models = {}
        self.feature_cols = []
        self.scaler = StandardScaler()
    
    def load_data(self, train_path='train.csv', test_path='test.csv', weather_path='wetter.csv', kiwo_path='kiwo.csv'):
        # Load main data
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        # Load auxiliary data
        self.weather_df = pd.read_csv(weather_path)
        self.kiwo_df = pd.read_csv(kiwo_path)
        
        # Convert dates
        self.train_df['Datum'] = pd.to_datetime(self.train_df['Datum'])
        self.test_df['Datum'] = pd.to_datetime(self.test_df['Datum'])
        self.weather_df['Datum'] = pd.to_datetime(self.weather_df['Datum'])
        self.kiwo_df['Datum'] = pd.to_datetime(self.kiwo_df['Datum'])
        
        # CRITICAL FIX: Filter test data to correct period (Aug 1 - Dec 28, 2018)
        print(f"\nOriginal test date range: {self.test_df['Datum'].min()} to {self.test_df['Datum'].max()}")
        self.test_df = self.test_df[(self.test_df['Datum'] >= '2018-08-01') & (self.test_df['Datum'] <= '2018-12-28')]
        print(f"Filtered test date range: {self.test_df['Datum'].min()} to {self.test_df['Datum'].max()}")
        
        
        return self
    
    def exploratory_analysis(self):
        # Product categories
        category_names = {1: 'Bread', 2: 'Rolls', 3: 'Croissant', 4: 'Confectionery', 5: 'Cake', 6: 'Seasonal Bread'}
        
        # Visualizations
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, category in enumerate(sorted(self.train_df['Warengruppe'].unique())):
            cat_data = self.train_df[self.train_df['Warengruppe'] == category]
            axes[i].plot(cat_data['Datum'], cat_data['Umsatz'], alpha=0.7)
            axes[i].set_title(f'{category_names[category]} (Category {category})', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Sales (Umsatz)')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sales_by_category.png', dpi=300, bbox_inches='tight')
        
        
        # Seasonality analysis
        self._plot_seasonality()
        
        return self
    
    def _plot_seasonality(self):
        temp_df = self.train_df.copy()
        temp_df['Year'] = temp_df['Datum'].dt.year
        temp_df['Month'] = temp_df['Datum'].dt.month
        temp_df['DayOfWeek'] = temp_df['Datum'].dt.dayofweek
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        # Monthly patterns
        monthly_sales = temp_df.groupby(['Month', 'Warengruppe'])['Umsatz'].mean().reset_index()
        for category in sorted(temp_df['Warengruppe'].unique()):
            cat_monthly = monthly_sales[monthly_sales['Warengruppe'] == category]
            axes[0].plot(cat_monthly['Month'], cat_monthly['Umsatz'], marker='o', label=f'Category {category}')
        axes[0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Average Sales')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Day of week patterns
        dow_sales = temp_df.groupby(['DayOfWeek', 'Warengruppe'])['Umsatz'].mean().reset_index()
        for category in sorted(temp_df['Warengruppe'].unique()):
            cat_dow = dow_sales[dow_sales['Warengruppe'] == category]
            axes[1].plot(cat_dow['DayOfWeek'], cat_dow['Umsatz'], marker='o', label=f'Category {category}')
        axes[1].set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day of Week (0=Monday)')
        axes[1].set_ylabel('Average Sales')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('seasonality_analysis.png', dpi=300, bbox_inches='tight')
       

    def feature_engineering(self, df):
        df = df.copy()
        
        # Temporal features
        df['Year'] = df['Datum'].dt.year
        df['Month'] = df['Datum'].dt.month
        df['Day'] = df['Datum'].dt.day
        df['DayOfWeek'] = df['Datum'].dt.dayofweek
        df['DayOfYear'] = df['Datum'].dt.dayofyear
        df['WeekOfYear'] = df['Datum'].dt.isocalendar().week
        df['Quarter'] = df['Datum'].dt.quarter
        
        # Cyclical encoding for time features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Weekend indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Month start/end indicators
        df['IsMonthStart'] = df['Datum'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Datum'].dt.is_month_end.astype(int)
        
        # Season
        df['Season'] = df['Month'] % 12 // 3 + 1
        
        # Merge weather data
        df = df.merge(self.weather_df, on='Datum', how='left')
        
        # Handle missing weather data
        df['Temperatur'] = df['Temperatur'].fillna(df.groupby('Month')['Temperatur'].transform('mean'))
        df['Windgeschwindigkeit'] = df['Windgeschwindigkeit'].fillna(df.groupby('Month')['Windgeschwindigkeit'].transform('mean'))
        df['Bewoelkung'] = df['Bewoelkung'].fillna(df['Bewoelkung'].median())
        df['Wettercode'] = df['Wettercode'].fillna(df['Wettercode'].mode()[0])
        
        # Kieler Woche indicator
        self.kiwo_df['KielerWoche'] = 1
        df = df.merge(self.kiwo_df[['Datum', 'KielerWoche']], on='Datum', how='left')
        df['KielerWoche'] = df['KielerWoche'].fillna(0).astype(int)
        
        # Days to/from Kieler Woche
        kiwo_dates = set(self.kiwo_df['Datum'].dt.date)
        df['DaysToKiwo'] = df['Datum'].apply(lambda x: min([abs((x.date() - kd).days) for kd in kiwo_dates] + [365]))
        
        # Holiday indicators
        df['IsHolidaySeason'] = ((df['Month'] == 12) & (df['Day'] >= 20) | (df['Month'] == 1) & (df['Day'] <= 6)).astype(int)
        df['IsEasterPeriod'] = ((df['Month'].isin([3, 4])) & (df['DayOfWeek'].isin([5, 6, 0]))).astype(int)
        
        # Product category
        df['Warengruppe_cat'] = df['Warengruppe'].astype(str)
        
        return df
    
    def create_lag_features(self, df, is_train=True):
        df = df.sort_values(['Warengruppe', 'Datum']).copy()
        
        if is_train:
            # Lag features
            for lag in [1, 2, 3, 7, 14, 21, 28]:
                df[f'Lag_{lag}'] = df.groupby('Warengruppe')['Umsatz'].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 28]:
                df[f'RollingMean_{window}'] = df.groupby('Warengruppe')['Umsatz'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                df[f'RollingStd_{window}'] = df.groupby('Warengruppe')['Umsatz'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
                df[f'RollingMin_{window}'] = df.groupby('Warengruppe')['Umsatz'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
                df[f'RollingMax_{window}'] = df.groupby('Warengruppe')['Umsatz'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
            
            # Exponential weighted moving average
            df['EWMA_7'] = df.groupby('Warengruppe')['Umsatz'].transform(lambda x: x.ewm(span=7, adjust=False).mean())
            df['EWMA_28'] = df.groupby('Warengruppe')['Umsatz'].transform(lambda x: x.ewm(span=28, adjust=False).mean())
            
            # Same day last week/month
            df['SameDayLastWeek'] = df.groupby('Warengruppe')['Umsatz'].shift(7)
            df['SameDayLastMonth'] = df.groupby('Warengruppe')['Umsatz'].shift(28)
        
        return df
    
    def prepare_data(self):
        # Engineer features
        print("Creating temporal and weather features...")
        train_feat = self.feature_engineering(self.train_df)
        test_feat = self.feature_engineering(self.test_df)
        
        # Create lag features
        print("Creating lag and rolling features...")
        train_feat = self.create_lag_features(train_feat, is_train=True)
        
        # For test set, combine with train to create lag features
        test_feat['Umsatz'] = np.nan
        combined = pd.concat([train_feat, test_feat], ignore_index=True)
        combined = combined.sort_values(['Warengruppe', 'Datum'])
        combined = self.create_lag_features(combined, is_train=True)
        
        # Split back
        train_feat = combined[combined['Umsatz'].notna()].copy()
        test_feat = combined[combined['Umsatz'].isna()].copy()
        test_feat = test_feat.drop('Umsatz', axis=1)
        
        # Fill remaining NaN values in lag features
        lag_cols = [col for col in train_feat.columns if 'Lag_' in col or 'Rolling' in col or 'EWMA' in col or 'SameDay' in col]
        
        for col in lag_cols:
            train_feat[col] = train_feat.groupby('Warengruppe')[col].fillna(method='bfill')
            train_feat[col] = train_feat.groupby('Warengruppe')[col].fillna(method='ffill')
            train_feat[col] = train_feat[col].fillna(train_feat[col].median())
            
            test_feat[col] = test_feat.groupby('Warengruppe')[col].fillna(method='bfill')
            test_feat[col] = test_feat.groupby('Warengruppe')[col].fillna(method='ffill')
            test_feat[col] = test_feat[col].fillna(train_feat[col].median())
        
        self.train_processed = train_feat
        self.test_processed = test_feat
        
        
        
        return self
    
    def train_models(self):

        # Define features
        exclude_cols = ['id', 'Datum', 'Umsatz', 'Warengruppe_cat']
        self.feature_cols = [col for col in self.train_processed.columns if col not in exclude_cols]
        
        X = self.train_processed[self.feature_cols]
        y = self.train_processed['Umsatz']
        
        # Handle any remaining NaN
        X = X.fillna(X.median())
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Models to train
        models = {
            'Ridge': Ridge(alpha=10),
            'Lasso': Lasso(alpha=1),
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                y_pred = np.maximum(y_pred, 0)
                
                # Calculate MAPE
                mape = np.mean(np.abs((y_val_fold - y_pred) / (y_val_fold + 1e-10))) * 100
                scores.append(mape)
            
            results[name] = {'mean_mape': np.mean(scores), 'std_mape': np.std(scores), 'scores': scores}
            
            print(f"  CV MAPE: {results[name]['mean_mape']:.2f}% (+/- {results[name]['std_mape']:.2f}%)")
            
            # Train on full dataset
            model.fit(X, y)
            self.models[name] = model
        
        # Display results summary
        results_df = pd.DataFrame({'Model': list(results.keys()), 'Mean MAPE': [results[m]['mean_mape'] for m in results.keys()], 'Std MAPE': [results[m]['std_mape'] for m in results.keys()]}).sort_values('Mean MAPE')
        
        self.model_results = results
        
        return self
    
    def predict(self, use_ensemble=True, weights=None):    
        X_test = self.test_processed[self.feature_cols]
        X_test = X_test.fillna(X_test.median())
        
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            pred = np.maximum(pred, 0)
            predictions[name] = pred
            
        
        if use_ensemble:
            if weights is None:
                # Weight by inverse MAPE
                mapes = [self.model_results[name]['mean_mape'] for name in self.models.keys()]
                weights = [1/m for m in mapes]
                weights = np.array(weights) / sum(weights)
            
            ensemble_pred = np.zeros(len(X_test))
            for i, name in enumerate(self.models.keys()):
                ensemble_pred += weights[i] * predictions[name]
                print(f"  {name}: weight = {weights[i]:.3f}")
            
            predictions['Ensemble'] = ensemble_pred
            
        self.predictions = predictions
        
        return self
    
    def create_submission(self, prediction_name='Ensemble', filename='submission.csv'):
        submission = pd.DataFrame({'id': self.test_processed['id'], 'umsatz': self.predictions[prediction_name]})
        
        submission.to_csv(filename, index=False)
        print(f"\nSubmission saved: {filename}")
        print(f"Shape: {submission.shape}")
        print(f"\nFirst few predictions:")
        print(submission.head(10))
        print(f"\nSummary statistics:")
        print(submission['umsatz'].describe())
        
        return submission
    
    def feature_importance_analysis(self):
        # Get feature importance from tree-based models
        importance_dict = {}
        
        for name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
            if name in self.models:
                if hasattr(self.models[name], 'feature_importances_'):
                    importance_dict[name] = self.models[name].feature_importances_
        
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=self.feature_cols)
            importance_df['Average'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('Average', ascending=False)
            
        
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))
            top_features = importance_df.head(20)
            top_features['Average'].plot(kind='barh', ax=ax)
            ax.set_title('Top 20 Feature Importances (Average across models)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        
        return self
    
    def run_complete_pipeline(self):
        
        # Execute pipeline
        self.load_data()
        self.exploratory_analysis()
        self.prepare_data()
        self.train_models()
        self.feature_importance_analysis()
        self.predict(use_ensemble=True)
        submission = self.create_submission()
        
    
        return submission


# Main execution
if __name__ == "__main__":
    forecaster = BakerySalesForecaster()
    submission = forecaster.run_complete_pipeline()
    
    # Create submission with Ridge ONLY (best CV score)
    submission_ridge = forecaster.create_submission(
    prediction_name='Ridge', 
    filename='submission_ridge.csv'
    )
    
    
    test = pd.read_csv('test.csv')
    test['Datum'] = pd.to_datetime(test['Datum'])
    test_filtered = test[(test['Datum'] >= '2018-08-01') & (test['Datum'] <= '2018-12-28')]
    
    check = test_filtered.merge(submission, on='id')
    