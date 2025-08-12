import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("H&M Markdown Prediction - Feature Engineering")
print("=" * 55)

# File paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

print("1. LOADING BASE DATASETS")
print("-" * 40)

# Load articles (product metadata)
print("Loading articles metadata...")
articles = pd.read_csv(raw_dir / "articles.csv")
print(f"✅ Articles: {len(articles):,} products with {articles.shape[1]} attributes")

# Load customers (for potential segmentation features)
print("Loading customer data...")
customers = pd.read_csv(raw_dir / "customers.csv")
print(f"✅ Customers: {len(customers):,} customers with {customers.shape[1]} attributes")

# Load sample transactions for feature engineering
print("Loading transaction sample for feature engineering...")
transactions_sample = pd.read_csv(
    raw_dir / "transactions_train.csv",
    nrows=500000,  # Sample for feature development
    parse_dates=['t_dat']
)
print(f"✅ Transaction sample: {len(transactions_sample):,} transactions")

print(f"\n2. PRODUCT FEATURES FROM ARTICLES")
print("-" * 40)

# Basic categorical features
categorical_features = [
    'product_type_name', 'product_group_name', 'department_name',
    'index_group_name', 'section_name', 'garment_group_name'
]

product_features = articles.copy()

# One-hot encode categorical features
print("Creating categorical features...")
for feature in categorical_features:
    if feature in product_features.columns:
        # Create dummy variables
        dummies = pd.get_dummies(product_features[feature], prefix=feature)
        product_features = pd.concat([product_features, dummies], axis=1)
        print(f"   {feature}: {dummies.shape[1]} categories")

# Color features
if 'colour_group_name' in product_features.columns:
    color_dummies = pd.get_dummies(product_features['colour_group_name'], prefix='color')
    product_features = pd.concat([product_features, color_dummies], axis=1)
    print(f"   Color groups: {color_dummies.shape[1]} colors")

# Text features from product descriptions
if 'detail_desc' in product_features.columns:
    print("Creating text features from descriptions...")
    # Fill missing descriptions
    product_features['detail_desc'] = product_features['detail_desc'].fillna('')
    
    # TF-IDF on product descriptions (top 50 terms)
    tfidf = TfidfVectorizer(max_features=50, stop_words='english', lowercase=True)
    tfidf_features = tfidf.fit_transform(product_features['detail_desc'])
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'desc_tfidf_{word}' for word in tfidf.get_feature_names_out()]
    )
    tfidf_df['article_id'] = product_features['article_id'].values
    
    product_features = pd.merge(product_features, tfidf_df, on='article_id', how='left')
    print(f"   Text features: {tfidf_df.shape[1]-1} TF-IDF terms")

print(f"✅ Product features created: {product_features.shape[1]} total features")

print(f"\n3. PRICING FEATURES")
print("-" * 40)

# Calculate pricing features per article
print("Computing pricing statistics per article...")
pricing_features = transactions_sample.groupby('article_id').agg({
    'price': [
        'mean', 'median', 'std', 'min', 'max', 'count',
        lambda x: x.quantile(0.25),  # Q1
        lambda x: x.quantile(0.75),  # Q3
    ],
    't_dat': ['min', 'max']  # First and last sale dates
}).round(2)

# Flatten column names
pricing_features.columns = [
    'price_mean', 'price_median', 'price_std', 'price_min', 'price_max', 
    'total_sales', 'price_q25', 'price_q75', 'first_sale_date', 'last_sale_date'
]
pricing_features = pricing_features.reset_index()

# Additional pricing features
pricing_features['price_range'] = pricing_features['price_max'] - pricing_features['price_min']
pricing_features['price_cv'] = pricing_features['price_std'] / pricing_features['price_mean']  # Coefficient of variation
pricing_features['price_volatility'] = pricing_features['price_std'] > 0  # Binary volatility indicator

# Days since first sale (product age)
max_date = transactions_sample['t_dat'].max()
pricing_features['days_since_launch'] = (max_date - pricing_features['first_sale_date']).dt.days
pricing_features['days_since_last_sale'] = (max_date - pricing_features['last_sale_date']).dt.days

print(f"✅ Pricing features: {len(pricing_features):,} articles with {pricing_features.shape[1]} pricing features")

print(f"\n4. DEMAND FEATURES")
print("-" * 40)

# Sales velocity and demand patterns
print("Computing demand and velocity features...")

# Weekly aggregation for velocity calculation
transactions_sample['week'] = transactions_sample['t_dat'].dt.isocalendar().week
transactions_sample['year_week'] = transactions_sample['t_dat'].dt.strftime('%Y-%W')

weekly_sales = transactions_sample.groupby(['article_id', 'year_week']).agg({
    'customer_id': ['count', 'nunique']  # transactions and unique customers per week
}).reset_index()
# Flatten multi-level columns
weekly_sales.columns = ['article_id', 'year_week', 'weekly_transactions', 'weekly_unique_customers']

# Calculate demand features per article
demand_features = weekly_sales.groupby('article_id').agg({
    'weekly_transactions': ['mean', 'std', 'max'],
    'weekly_unique_customers': ['mean', 'std', 'max']
}).round(2)

demand_features.columns = [
    'avg_weekly_sales', 'std_weekly_sales', 'max_weekly_sales',
    'avg_weekly_customers', 'std_weekly_customers', 'max_weekly_customers'
]
demand_features = demand_features.reset_index()

# Sales momentum (recent vs overall average)
recent_weeks = 4
recent_data = transactions_sample[
    transactions_sample['t_dat'] >= (transactions_sample['t_dat'].max() - timedelta(weeks=recent_weeks))
]

recent_demand = recent_data.groupby('article_id').agg({
    'customer_id': 'count'
}).reset_index()
recent_demand.columns = ['article_id', 'recent_sales']

demand_features = pd.merge(demand_features, recent_demand, on='article_id', how='left')
demand_features['recent_sales'] = demand_features['recent_sales'].fillna(0)
demand_features['sales_momentum'] = demand_features['recent_sales'] / (demand_features['avg_weekly_sales'] * recent_weeks)
demand_features['sales_momentum'] = demand_features['sales_momentum'].fillna(0)

print(f"✅ Demand features: {len(demand_features):,} articles with {demand_features.shape[1]} demand features")

print(f"\n5. CALENDAR FEATURES")
print("-" * 40)

# Create calendar features for time-based patterns
def create_calendar_features(date_col):
    """Create calendar and seasonality features"""
    features = pd.DataFrame()
    features['year'] = date_col.dt.year
    features['month'] = date_col.dt.month
    features['quarter'] = date_col.dt.quarter
    features['day_of_year'] = date_col.dt.dayofyear
    features['week_of_year'] = date_col.dt.isocalendar().week
    features['day_of_week'] = date_col.dt.dayofweek
    features['is_weekend'] = date_col.dt.dayofweek >= 5
    
    # Holiday proximity (simplified)
    features['is_december'] = date_col.dt.month == 12  # Holiday season
    features['is_january'] = date_col.dt.month == 1    # Post-holiday
    features['is_summer'] = date_col.dt.month.isin([6, 7, 8])  # Summer season
    
    return features

# We'll add these features when we create training instances
print("✅ Calendar feature functions ready")

print(f"\n6. COMBINING ALL FEATURES")
print("-" * 40)

# Merge all feature sets
print("Combining product, pricing, and demand features...")

# Start with product features
feature_dataset = product_features.copy()

# Add pricing features
feature_dataset = pd.merge(feature_dataset, pricing_features, on='article_id', how='left')
print(f"   After adding pricing: {feature_dataset.shape}")

# Add demand features
feature_dataset = pd.merge(feature_dataset, demand_features, on='article_id', how='left')
print(f"   After adding demand: {feature_dataset.shape}")

# Fill missing values
numeric_columns = feature_dataset.select_dtypes(include=[np.number]).columns
feature_dataset[numeric_columns] = feature_dataset[numeric_columns].fillna(0)

print(f"✅ Combined feature dataset: {feature_dataset.shape[0]:,} articles × {feature_dataset.shape[1]} features")

print(f"\n7. FEATURE IMPORTANCE PREVIEW")
print("-" * 40)

# Show key feature categories
feature_types = {
    'Product Categories': len([col for col in feature_dataset.columns if any(cat in col for cat in categorical_features)]),
    'Color Features': len([col for col in feature_dataset.columns if 'color_' in col]),
    'Text Features': len([col for col in feature_dataset.columns if 'desc_tfidf_' in col]),
    'Pricing Features': len([col for col in feature_dataset.columns if 'price_' in col]),
    'Demand Features': len([col for col in feature_dataset.columns if any(x in col for x in ['sales', 'customers', 'momentum'])]),
    'Temporal Features': len([col for col in feature_dataset.columns if any(x in col for x in ['days_', 'first_', 'last_'])])
}

for feature_type, count in feature_types.items():
    print(f"   {feature_type}: {count} features")

print(f"\n8. SAVING ENGINEERED FEATURES")
print("-" * 40)

# Save feature dataset
features_path = processed_dir / "engineered_features.csv"
feature_dataset.to_csv(features_path, index=False)
print(f"✅ Saved feature dataset: {features_path}")

# Save feature metadata
feature_metadata = {
    'total_features': feature_dataset.shape[1],
    'feature_types': feature_types,
    'articles_covered': feature_dataset.shape[0],
    'key_features': {
        'categorical': categorical_features,
        'pricing': ['price_mean', 'price_std', 'price_volatility', 'days_since_launch'],
        'demand': ['avg_weekly_sales', 'sales_momentum', 'recent_sales'],
        'product': ['product_type_name', 'colour_group_name', 'department_name']
    }
}

import json
metadata_path = processed_dir / "feature_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(feature_metadata, f, indent=2, default=str)

print(f"✅ Saved feature metadata: {metadata_path}")

print(f"\n9. SAMPLE FEATURE PREVIEW")
print("-" * 40)

# Show sample of engineered features
sample_features = feature_dataset.head(3)
key_columns = ['article_id', 'price_mean', 'price_volatility', 'avg_weekly_sales', 'sales_momentum', 'days_since_launch']
available_columns = [col for col in key_columns if col in feature_dataset.columns]

print("Sample engineered features:")
print(sample_features[available_columns].to_string())

print(f"\n" + "=" * 55)
print("✅ FEATURE ENGINEERING COMPLETE!")
print(f"Ready for model training with {feature_dataset.shape[1]} features")
print("=" * 55)
