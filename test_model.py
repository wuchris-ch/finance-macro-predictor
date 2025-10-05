"""Quick test script to verify model training and get results"""

from ml_model import M2Predictor
import pandas as pd
import sys

try:
    print("Starting model test...", flush=True)
    
    # Load data
    df = pd.read_csv('M2SL.csv')
    print(f"Loaded {len(df)} observations", flush=True)
    
    # Initialize and train
    predictor = M2Predictor(n_estimators=100, max_depth=10, random_state=42)
    df_features = predictor.create_features(df)
    print(f"Created features", flush=True)
    
    X_train, X_test, y_train, y_test, train_dates, test_dates = predictor.prepare_data(df_features, test_months=24)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}", flush=True)
    
    predictor.train(X_train, y_train)
    print("Training complete", flush=True)
    
    metrics, y_pred = predictor.evaluate(X_test, y_test)
    print(f"\nMetrics:", flush=True)
    print(f"RMSE: {metrics['rmse']:.2f}", flush=True)
    print(f"MAE: {metrics['mae']:.2f}", flush=True)
    print(f"R²: {metrics['r2']:.4f}", flush=True)
    print(f"MAPE: {metrics['mape']:.2f}%", flush=True)
    
    # Save model
    predictor.save_model('m2_model.pkl')
    
    # Predict future
    future = predictor.predict_future(df, n_months=12)
    print(f"\nFuture predictions:", flush=True)
    for idx, row in future.head(12).iterrows():
        print(f"{row['observation_date'].strftime('%Y-%m')}: ${row['predicted_M2SL']:.2f}B", flush=True)
    
    future.to_csv('future_predictions.csv', index=False)
    print("\nSaved predictions to future_predictions.csv", flush=True)
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)