import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("H&M Dataset Initial Exploration")
print("=" * 50)

# File paths
articles_path = '../data/raw/articles.csv'
customers_path = '../data/raw/customers.csv'
transactions_path = '../data/raw/transactions_train.csv'

print("\n1. LOADING DATASETS")
print("-" * 30)

# Load articles (product metadata)
print("Loading articles.csv...")
articles = pd.read_csv(articles_path)
print(f"✅ Articles loaded: {articles.shape[0]:,} products, {articles.shape[1]} columns")

# Load customers  
print("Loading customers.csv...")
customers = pd.read_csv(customers_path)
print(f"✅ Customers loaded: {customers.shape[0]:,} customers, {customers.shape[1]} columns")

# Load transactions (this is the big one - 3.5GB)
print("Loading transactions_train.csv... (this may take a moment)")
transactions = pd.read_csv(transactions_path, parse_dates=['t_dat'])
print(f"✅ Transactions loaded: {transactions.shape[0]:,} transactions, {transactions.shape[1]} columns")

print("\n2. DATASET OVERVIEW")
print("-" * 30)

# Articles overview
print(f"\n📦 ARTICLES ({articles.shape[0]:,} products)")
print(f"Columns: {list(articles.columns)}")
print(f"Memory usage: {articles.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Customers overview  
print(f"\n👥 CUSTOMERS ({customers.shape[0]:,} customers)")
print(f"Columns: {list(customers.columns)}")
print(f"Memory usage: {customers.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Transactions overview
print(f"\n🛒 TRANSACTIONS ({transactions.shape[0]:,} transactions)")
print(f"Columns: {list(transactions.columns)}")
print(f"Memory usage: {transactions.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"Date range: {transactions['t_dat'].min()} to {transactions['t_dat'].max()}")
print(f"Time span: {(transactions['t_dat'].max() - transactions['t_dat'].min()).days} days")

print("\n3. KEY STATISTICS")
print("-" * 30)

# Unique counts
print(f"Unique articles in catalog: {articles['article_id'].nunique():,}")
print(f"Unique customers: {customers['customer_id'].nunique():,}")
print(f"Unique articles sold: {transactions['article_id'].nunique():,}")
print(f"Unique customers who purchased: {transactions['customer_id'].nunique():,}")

# Coverage analysis
articles_sold_pct = (transactions['article_id'].nunique() / articles['article_id'].nunique()) * 100
customers_active_pct = (transactions['customer_id'].nunique() / customers['customer_id'].nunique()) * 100
print(f"Articles sold vs catalog: {articles_sold_pct:.1f}%")
print(f"Active customers: {customers_active_pct:.1f}%")

print("\n4. PRICING ANALYSIS")
print("-" * 30)

# Price statistics
print(f"Price statistics:")
print(f"  Min price: ${transactions['price'].min():.2f}")
print(f"  Max price: ${transactions['price'].max():.2f}")
print(f"  Mean price: ${transactions['price'].mean():.2f}")
print(f"  Median price: ${transactions['price'].median():.2f}")

# Price distribution by percentiles
price_percentiles = [10, 25, 50, 75, 90, 95, 99]
print(f"\nPrice percentiles:")
for p in price_percentiles:
    value = np.percentile(transactions['price'], p)
    print(f"  {p}th percentile: ${value:.2f}")

print("\n5. TEMPORAL PATTERNS")
print("-" * 30)

# Monthly transaction volume
monthly_volume = transactions.groupby(transactions['t_dat'].dt.to_period('M')).size()
print(f"Monthly transaction volume (last 6 months):")
for month, volume in monthly_volume.tail(6).items():
    print(f"  {month}: {volume:,} transactions")

print("\n6. PRODUCT CATEGORIES")
print("-" * 30)

# Top product types
if 'product_type_name' in articles.columns:
    top_products = articles['product_type_name'].value_counts().head(10)
    print("Top 10 product types:")
    for product, count in top_products.items():
        print(f"  {product}: {count:,} SKUs")

print("\n7. MISSING VALUES CHECK")
print("-" * 30)

def check_missing(df, name):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print(f"\n{name} missing values:")
    for col in missing.index:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")
        
check_missing(articles, "ARTICLES")
check_missing(customers, "CUSTOMERS") 
check_missing(transactions, "TRANSACTIONS")

print("\n8. DATA QUALITY INSIGHTS")
print("-" * 30)

# Duplicate transactions
dup_transactions = transactions.duplicated().sum()
print(f"Duplicate transactions: {dup_transactions:,}")

# Articles with price history (potential for markdown analysis)
articles_with_multiple_prices = transactions.groupby('article_id')['price'].nunique()
articles_price_changes = (articles_with_multiple_prices > 1).sum()
print(f"Articles with price variations: {articles_price_changes:,} ({(articles_price_changes/articles_with_multiple_prices.count())*100:.1f}%)")

# Transaction frequency
print(f"Average transactions per customer: {transactions.shape[0] / transactions['customer_id'].nunique():.1f}")
print(f"Average transactions per article: {transactions.shape[0] / transactions['article_id'].nunique():.1f}")

print("\n9. SAMPLE DATA PREVIEW")
print("-" * 30)

print("\nArticles sample:")
print(articles.head(3).to_string())

print("\nCustomers sample:")
print(customers.head(3).to_string())

print("\nTransactions sample:")
print(transactions.head(3).to_string())

print("\n" + "=" * 50)
print("✅ Initial data exploration complete!")
print("Next steps: Markdown label creation and feature engineering")
print("=" * 50)
