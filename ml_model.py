"""
M2 Money Supply Prediction Model
Uses Random Forest Regressor to predict future M2 values
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class M2Predictor:
    """M2 Money Supply Prediction Model using Random Forest"""
    
    def __init__(self, n_estimators=200, max_depth=15, random_state=42):
        """
        Initialize the M2 Predictor
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.last_date = None
        self.last_values = None
        
    def create_features(self, df):
        """
        Create lag features, growth rates, rolling statistics, and temporal features
        
        Args:
            df: DataFrame with 'observation_date' and 'M2SL' columns
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df = df.sort_values('observation_date').reset_index(drop=True)
        
        # Lag features (1, 3, 6, 12 months)
        df['lag_1'] = df['M2SL'].shift(1)
        df['lag_3'] = df['M2SL'].shift(3)
        df['lag_6'] = df['M2SL'].shift(6)
        df['lag_12'] = df['M2SL'].shift(12)
        
        # Growth rates
        df['mom_growth'] = df['M2SL'].pct_change(1) * 100  # Month-over-month
        df['yoy_growth'] = df['M2SL'].pct_change(12) * 100  # Year-over-year
        df['qoq_growth'] = df['M2SL'].pct_change(3) * 100  # Quarter-over-quarter
        
        # Rolling statistics (moving averages and volatility)
        df['ma_3'] = df['M2SL'].rolling(window=3).mean()
        df['ma_6'] = df['M2SL'].rolling(window=6).mean()
        df['ma_12'] = df['M2SL'].rolling(window=12).mean()
        df['std_3'] = df['M2SL'].rolling(window=3).std()
        df['std_6'] = df['M2SL'].rolling(window=6).std()
        df['std_12'] = df['M2SL'].rolling(window=12).std()
        
        # Temporal features
        df['month'] = df['observation_date'].dt.month
        df['quarter'] = df['observation_date'].dt.quarter
        df['year'] = df['observation_date'].dt.year
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        # Trend features
        df['trend'] = np.arange(len(df))
        df['trend_normalized'] = df['trend'] / len(df)
        
        # Cyclical features (encode month as sine/cosine)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def prepare_data(self, df, test_months=24):
        """
        Prepare training and test datasets
        
        Args:
            df: DataFrame with features
            test_months: Number of months to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test, train_dates, test_dates
        """
        # Drop rows with NaN values (from lag features)
        df_clean = df.dropna().reset_index(drop=True)
        
        # Define feature columns (exclude date and target)
        feature_cols = [col for col in df_clean.columns 
                       if col not in ['observation_date', 'M2SL']]
        
        # Split into train and test
        split_idx = len(df_clean) - test_months
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df['M2SL']
        X_test = test_df[feature_cols]
        y_test = test_df['M2SL']
        
        train_dates = train_df['observation_date']
        test_dates = test_df['observation_date']
        
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with performance metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics, y_pred
    
    def predict_future(self, df, n_months=12):
        """
        Predict future M2 values
        
        Args:
            df: Historical DataFrame with features
            n_months: Number of months to predict
            
        Returns:
            DataFrame with predictions
        """
        df_with_features = self.create_features(df)
        df_clean = df_with_features.dropna().reset_index(drop=True)
        
        # Store last known values for iterative prediction
        last_row = df_clean.iloc[-1].copy()
        last_date = pd.to_datetime(last_row['observation_date'])
        
        predictions = []
        
        for i in range(n_months):
            # Create next month's date
            next_date = last_date + pd.DateOffset(months=i+1)
            
            # Prepare features for prediction
            next_features = {}
            
            # Use last known values for lags
            if i == 0:
                next_features['lag_1'] = last_row['M2SL']
                next_features['lag_3'] = df_clean.iloc[-3]['M2SL'] if len(df_clean) >= 3 else last_row['M2SL']
                next_features['lag_6'] = df_clean.iloc[-6]['M2SL'] if len(df_clean) >= 6 else last_row['M2SL']
                next_features['lag_12'] = df_clean.iloc[-12]['M2SL'] if len(df_clean) >= 12 else last_row['M2SL']
            else:
                # Use previous predictions for lags
                next_features['lag_1'] = predictions[i-1]['predicted_M2SL']
                next_features['lag_3'] = predictions[i-3]['predicted_M2SL'] if i >= 3 else df_clean.iloc[-(3-i)]['M2SL']
                next_features['lag_6'] = predictions[i-6]['predicted_M2SL'] if i >= 6 else df_clean.iloc[-(6-i)]['M2SL']
                next_features['lag_12'] = predictions[i-12]['predicted_M2SL'] if i >= 12 else df_clean.iloc[-(12-i)]['M2SL']
            
            # Calculate growth rates based on lags
            next_features['mom_growth'] = ((next_features['lag_1'] - 
                                           (predictions[i-2]['predicted_M2SL'] if i >= 2 else df_clean.iloc[-2]['M2SL'])) / 
                                          (predictions[i-2]['predicted_M2SL'] if i >= 2 else df_clean.iloc[-2]['M2SL'])) * 100
            
            next_features['yoy_growth'] = ((next_features['lag_1'] - next_features['lag_12']) / 
                                          next_features['lag_12']) * 100
            
            next_features['qoq_growth'] = ((next_features['lag_1'] - next_features['lag_3']) / 
                                          next_features['lag_3']) * 100
            
            # Rolling averages (approximate using available data)
            recent_values = [predictions[j]['predicted_M2SL'] for j in range(max(0, i-3), i)]
            recent_values.append(next_features['lag_1'])
            next_features['ma_3'] = np.mean(recent_values[-3:])
            next_features['ma_6'] = np.mean(recent_values[-6:]) if len(recent_values) >= 6 else next_features['ma_3']
            next_features['ma_12'] = np.mean(recent_values[-12:]) if len(recent_values) >= 12 else next_features['ma_6']
            
            next_features['std_3'] = np.std(recent_values[-3:]) if len(recent_values) >= 3 else last_row['std_3']
            next_features['std_6'] = np.std(recent_values[-6:]) if len(recent_values) >= 6 else last_row['std_6']
            next_features['std_12'] = np.std(recent_values[-12:]) if len(recent_values) >= 12 else last_row['std_12']
            
            # Temporal features
            next_features['month'] = next_date.month
            next_features['quarter'] = next_date.quarter
            next_features['year'] = next_date.year
            next_features['year_normalized'] = (next_features['year'] - df_clean['year'].min()) / (df_clean['year'].max() - df_clean['year'].min())
            
            # Trend features
            next_features['trend'] = last_row['trend'] + i + 1
            next_features['trend_normalized'] = next_features['trend'] / (len(df_clean) + n_months)
            
            # Cyclical features
            next_features['month_sin'] = np.sin(2 * np.pi * next_features['month'] / 12)
            next_features['month_cos'] = np.cos(2 * np.pi * next_features['month'] / 12)
            
            # Create feature vector in correct order
            X_next = np.array([[next_features[col] for col in self.feature_names]])
            
            # Scale and predict
            X_next_scaled = self.scaler.transform(X_next)
            pred = self.model.predict(X_next_scaled)[0]
            
            predictions.append({
                'observation_date': next_date,
                'predicted_M2SL': pred
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath='m2_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='m2_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")


def main():
    """Main training and evaluation pipeline"""
    
    print("=" * 60)
    print("M2 Money Supply Prediction Model")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('M2SL.csv')
    print(f"   Loaded {len(df)} observations from {df['observation_date'].min()} to {df['observation_date'].max()}")
    
    # Initialize predictor
    predictor = M2Predictor(n_estimators=200, max_depth=15, random_state=42)
    
    # Create features
    print("\n2. Creating features...")
    df_features = predictor.create_features(df)
    print(f"   Created {len([col for col in df_features.columns if col not in ['observation_date', 'M2SL']])} features")
    
    # Prepare data
    print("\n3. Preparing train/test split (last 24 months for testing)...")
    X_train, X_test, y_train, y_test, train_dates, test_dates = predictor.prepare_data(df_features, test_months=24)
    print(f"   Training set: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
    print(f"   Test set: {len(X_test)} samples ({test_dates.min()} to {test_dates.max()})")
    
    # Train model
    print("\n4. Training Random Forest model...")
    predictor.train(X_train, y_train)
    
    # Evaluate on test set
    print("\n5. Evaluating model performance...")
    metrics, y_pred = predictor.evaluate(X_test, y_test)
    
    print("\n   Test Set Performance Metrics:")
    print(f"   - RMSE: ${metrics['rmse']:.2f} billion")
    print(f"   - MAE:  ${metrics['mae']:.2f} billion")
    print(f"   - R²:   {metrics['r2']:.4f}")
    print(f"   - MAPE: {metrics['mape']:.2f}%")
    
    # Feature importance
    print("\n6. Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': predictor.feature_names,
        'importance': predictor.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    # Save model
    print("\n7. Saving model...")
    predictor.save_model('m2_model.pkl')
    
    # Generate future predictions
    print("\n8. Generating predictions for next 12 months...")
    future_predictions = predictor.predict_future(df, n_months=12)
    
    print("\n   Future M2 Predictions:")
    print("   " + "-" * 50)
    for idx, row in future_predictions.iterrows():
        print(f"   {row['observation_date'].strftime('%Y-%m-%d')}: ${row['predicted_M2SL']:.2f} billion")
    
    # Save predictions to CSV
    future_predictions.to_csv('future_predictions.csv', index=False)
    print("\n   Predictions saved to future_predictions.csv")
    
    # Create test predictions DataFrame for analysis
    test_results = pd.DataFrame({
        'observation_date': test_dates.values,
        'actual_M2SL': y_test.values,
        'predicted_M2SL': y_pred
    })
    test_results.to_csv('test_predictions.csv', index=False)
    print("   Test results saved to test_predictions.csv")
    
    print("\n" + "=" * 60)
    print("Model training and evaluation complete!")
    print("=" * 60)
    
    return predictor, metrics, future_predictions


if __name__ == "__main__":
    predictor, metrics, predictions = main()