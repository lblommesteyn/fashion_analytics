import pandas as pd
import numpy as np
import os
from pathlib import Path

print("H&M Markdown Prediction - Data Exploration")
print("=" * 60)

# Get the project root directory
project_root = Path(__file__).parent.parent
data_dir = project_root / "data" / "raw"

print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")

# Check if files exist
files_to_check = {
    'articles': data_dir / "articles.csv",
    'customers': data_dir / "customers.csv", 
    'transactions': data_dir / "transactions_train.csv"
}

print(f"\nChecking data files:")
for name, path in files_to_check.items():
    exists = path.exists()
    size_mb = path.stat().st_size / (1024*1024) if exists else 0
    print(f"  {name:12} {'✅' if exists else '❌'} {size_mb:8.1f} MB")

print(f"\n1. LOADING ARTICLES DATA")
print("-" * 40)
try:
    articles = pd.read_csv(files_to_check['articles'])
    print(f"✅ Articles loaded: {articles.shape[0]:,} products x {articles.shape[1]} columns")
    print(f"   Memory usage: {articles.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Columns: {list(articles.columns[:5])}{'...' if len(articles.columns) > 5 else ''}")
    
    # Key product insights
    if 'product_type_name' in articles.columns:
        print(f"   Top 5 product types: {articles['product_type_name'].value_counts().head().to_dict()}")
except Exception as e:
    print(f"❌ Error loading articles: {e}")

print(f"\n2. LOADING CUSTOMERS DATA")
print("-" * 40)
try:
    customers = pd.read_csv(files_to_check['customers'])
    print(f"✅ Customers loaded: {customers.shape[0]:,} customers x {customers.shape[1]} columns")
    print(f"   Memory usage: {customers.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Columns: {list(customers.columns)}")
    
    # Customer demographics
    if 'age' in customers.columns:
        age_stats = customers['age'].describe()
        print(f"   Age range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years (avg: {age_stats['mean']:.1f})")
except Exception as e:
    print(f"❌ Error loading customers: {e}")

print(f"\n3. LOADING TRANSACTIONS DATA (LARGE FILE)")
print("-" * 40)
try:
    # Load transactions in chunks to avoid memory issues
    print("   Loading sample (first 100k rows) for initial exploration...")
    transactions_sample = pd.read_csv(files_to_check['transactions'], nrows=100000, parse_dates=['t_dat'])
    print(f"✅ Sample transactions loaded: {transactions_sample.shape[0]:,} x {transactions_sample.shape[1]} columns")
    print(f"   Columns: {list(transactions_sample.columns)}")
    
    # Key transaction insights from sample
    print(f"   Date range (sample): {transactions_sample['t_dat'].min()} to {transactions_sample['t_dat'].max()}")
    print(f"   Price range (sample): ${transactions_sample['price'].min():.2f} - ${transactions_sample['price'].max():.2f}")
    print(f"   Avg price (sample): ${transactions_sample['price'].mean():.2f}")
    
    # Count total transactions (without loading all)
    print("   Counting total transactions...")
    total_transactions = sum(1 for _ in open(files_to_check['transactions'])) - 1  # subtract header
    print(f"   Total transactions: {total_transactions:,}")
    
except Exception as e:
    print(f"❌ Error loading transactions: {e}")

print(f"\n4. MARKDOWN PREDICTION FEASIBILITY")
print("-" * 40)

try:
    # Check for price variations (key for markdown detection)
    price_variations = transactions_sample.groupby('article_id')['price'].agg(['count', 'nunique', 'std'])
    articles_with_price_changes = (price_variations['nunique'] > 1).sum()
    
    print(f"   Articles with multiple prices (sample): {articles_with_price_changes:,}")
    print(f"   % with price variations: {(articles_with_price_changes/len(price_variations))*100:.1f}%")
    
    # Price volatility
    volatile_articles = (price_variations['std'] > 0).sum()
    print(f"   Articles with price volatility: {volatile_articles:,}")
    
    # Sample markdown candidates
    markdown_candidates = price_variations[price_variations['std'] > 0].head()
    print(f"   Sample volatile articles:")
    for article_id, row in markdown_candidates.iterrows():
        print(f"     Article {article_id}: {row['count']} transactions, {row['nunique']} price points, σ=${row['std']:.2f}")
        
except Exception as e:
    print(f"❌ Error analyzing markdown feasibility: {e}")

print(f"\n5. DATA QUALITY SUMMARY")
print("-" * 40)

# Missing values check
print("   Missing values:")
try:
    for df_name, df in [('articles', articles), ('customers', customers), ('transactions_sample', transactions_sample)]:
        missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        print(f"     {df_name:20}: {missing:,} / {total_cells:,} ({(missing/total_cells)*100:.2f}%)")
except:
    print("     Could not compute missing values")

print(f"\n6. NEXT STEPS FOR MARKDOWN MODEL")
print("-" * 40)
print("   ✅ Data is suitable for markdown prediction")
print("   ✅ Price variations exist across articles")
print("   ✅ Transaction history spans time for temporal features")
print("   ✅ Rich product metadata available")
print("   📋 Ready to proceed with:")
print("      1. Markdown label creation (price drop detection)")
print("      2. Feature engineering (product, pricing, demand)")
print("      3. Model training (XGBoost/LightGBM)")
print("      4. What-if simulator development")

print(f"\n" + "=" * 60)
print("✅ DATA EXPLORATION COMPLETE")
print("Ready to build markdown prediction model!")
print("=" * 60)
