from datasets import load_dataset
import pandas as pd

try:
    print("Loading dataset...")
    ds = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF")
    print("Dataset loaded.")
    print(ds)
    
    # Convert to pandas to check head and columns
    df = ds['train'].to_pandas()
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nClass distribution:\n", df.iloc[:, -1].value_counts()) # Assuming last column is target due to name usually
    
except Exception as e:
    print(f"Error loading dataset: {e}")
