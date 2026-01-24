import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Configuration
DATASET_NAME = "CiferAI/Cifer-Fraud-Detection-Dataset-AF"
TARGET_COL = "isFraud"
DROP_COLS = ["nameOrig", "nameDest", "isFlaggedFraud"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_preprocess():
    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME, split="train") 
    
    # Load separate split or subset if practical for debugging, but let's stick to full
    print("Converting to Pandas...")
    df = ds.to_pandas()
    
    # Preprocessing
    print("Preprocessing...")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns to encode: {cat_cols}")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test

def train_lightgbm(X_train, y_train, device='cpu'):
    print(f"\nScanning for {device.upper()} training...")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'dart',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'bagging_fraction': 1,
        'feature_fraction': 1,
        'random_state': RANDOM_STATE,
        'verbose': 1 # Enable verbose
    }
    
    if device == 'gpu':
        params['device'] = 'gpu'
    
    start_time = time.time()
    try:
        # Use class_weight instead of is_unbalance
        model = lgb.LGBMClassifier(class_weight='balanced', **params)
        model.fit(X_train, y_train)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{device.upper()} training completed in {duration:.4f} seconds")
        return model, duration
    except Exception as e:
        print(f"Failed to train on {device.upper()}: {e}")
        return None, None

def evaluate_model(model, X_test, y_test, device_name):
    if model is None:
        return {}
    
    print(f"\nEvaluating {device_name} model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Debug predictions
    print(f"Unique predicted probabilities (first 10): {np.unique(y_pred_proba)[:10]}")
    print(f"Unique predictions: {np.unique(y_pred)}")
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Results for {device_name}:")
    print(f"  AUC: {auc:.4f}")
    
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model,
        'auc': auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }

def plot_results(results_cpu, results_gpu, y_test):
    print("\nGenerating plots...")
    
    # ROC Curve
    plt.figure(figsize=(10, 6))
    if 'model' in results_cpu:
        fpr, tpr, _ = roc_curve(y_test, results_cpu['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"CPU (AUC = {results_cpu['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Confusion Matrix CPU
    if 'model' in results_cpu:
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, results_cpu['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.savefig('confusion_matrix_cpu.png')
        plt.close()

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Train CPU
    model_cpu, time_cpu = train_lightgbm(X_train, y_train, device='cpu')
    results_cpu = evaluate_model(model_cpu, X_test, y_test, "CPU")
    if model_cpu:
        results_cpu['duration'] = time_cpu
    
    # Train GPU (will fail)
    model_gpu, time_gpu = train_lightgbm(X_train, y_train, device='gpu')
    results_gpu = evaluate_model(model_gpu, X_test, y_test, "GPU")
    if model_gpu:
        results_gpu['duration'] = time_gpu
        
    plot_results(results_cpu, results_gpu, y_test)
    print("\nDone!")

if __name__ == "__main__":
    main()
