import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("H&M Markdown Label Creation Pipeline")
print("=" * 50)

# Configuration
MARKDOWN_THRESHOLD = 0.15  # 15% price drop threshold
ROLLING_WINDOW_DAYS = 14   # Rolling median window
PREDICTION_HORIZON_DAYS = 30  # Predict markdown in next 30 days

# File paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

print(f"Loading transaction data...")
print(f"This may take a few moments for the large file (3.5GB)...")

# Load transactions with optimized dtypes for memory efficiency
dtype_dict = {
    'customer_id': 'category',
    'article_id': 'int32',
    'price': 'float32'
}

transactions = pd.read_csv(
    raw_dir / "transactions_train.csv",
    dtype=dtype_dict,
    parse_dates=['t_dat']
)

print(f"✅ Loaded {len(transactions):,} transactions")
print(f"   Date range: {transactions['t_dat'].min()} to {transactions['t_dat'].max()}")
print(f"   Memory usage: {transactions.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Sort by article and date for efficient processing
print(f"\nSorting transactions by article and date...")
transactions = transactions.sort_values(['article_id', 't_dat'])

print(f"\n1. CREATING DAILY PRICE AGGREGATIONS")
print("-" * 40)

# Create daily price summary per article
daily_prices = transactions.groupby(['article_id', 't_dat']).agg({
    'price': ['mean', 'min', 'max', 'count'],
    'customer_id': 'nunique'
}).round(2)

# Flatten column names
daily_prices.columns = ['avg_price', 'min_price', 'max_price', 'transactions', 'unique_customers']
daily_prices = daily_prices.reset_index()

print(f"✅ Created daily price aggregations: {len(daily_prices):,} article-day records")

print(f"\n2. CALCULATING ROLLING PRICE BASELINES")
print("-" * 40)

# Calculate rolling median price for each article
def calculate_rolling_baseline(df):
    """Calculate rolling median price baseline for markdown detection"""
    df = df.sort_values('t_dat')
    df['rolling_median_price'] = df['avg_price'].rolling(
        window=ROLLING_WINDOW_DAYS, 
        min_periods=7  # Need at least 7 days of data
    ).median()
    return df

# Apply rolling calculation per article
daily_prices = daily_prices.groupby('article_id').apply(calculate_rolling_baseline).reset_index(drop=True)

print(f"✅ Calculated {ROLLING_WINDOW_DAYS}-day rolling median baselines")

print(f"\n3. DETECTING MARKDOWN EVENTS")
print("-" * 40)

# Calculate price drops vs baseline
daily_prices['price_drop_pct'] = (
    (daily_prices['rolling_median_price'] - daily_prices['avg_price']) / 
    daily_prices['rolling_median_price']
)

# Mark markdown events (price drop >= threshold)
daily_prices['is_markdown'] = (daily_prices['price_drop_pct'] >= MARKDOWN_THRESHOLD) & \
                              (daily_prices['rolling_median_price'].notna())

# Count markdown events
markdown_events = daily_prices['is_markdown'].sum()
total_valid_days = daily_prices['rolling_median_price'].notna().sum()
markdown_rate = (markdown_events / total_valid_days) * 100

print(f"✅ Detected markdown events:")
print(f"   Total markdown days: {markdown_events:,}")
print(f"   Total valid observation days: {total_valid_days:,}")
print(f"   Markdown rate: {markdown_rate:.2f}%")
print(f"   Threshold used: {MARKDOWN_THRESHOLD*100}% price drop")

print(f"\n4. CREATING PREDICTION LABELS")
print("-" * 40)

# Create forward-looking labels: "Will this article be marked down in next 30 days?"
def create_prediction_labels(df):
    """Create binary labels for markdown prediction"""
    df = df.sort_values('t_dat')
    
    # For each day, check if there's a markdown in the next N days
    df['markdown_next_30d'] = False
    
    for i in range(len(df)):
        current_date = df.iloc[i]['t_dat']
        future_date = current_date + timedelta(days=PREDICTION_HORIZON_DAYS)
        
        # Check if any markdown occurs in the next 30 days
        future_markdowns = df[
            (df['t_dat'] > current_date) & 
            (df['t_dat'] <= future_date) & 
            (df['is_markdown'] == True)
        ]
        
        if len(future_markdowns) > 0:
            df.iloc[i, df.columns.get_loc('markdown_next_30d')] = True
    
    return df

print(f"   Creating {PREDICTION_HORIZON_DAYS}-day forward-looking labels...")
print(f"   Processing per article (this may take a moment)...")

# Apply label creation per article (sample on subset for demo)
print(f"   Processing top 1000 articles for label creation demo...")
top_articles = daily_prices['article_id'].value_counts().head(1000).index
subset_data = daily_prices[daily_prices['article_id'].isin(top_articles)].copy()

labeled_data = subset_data.groupby('article_id').apply(create_prediction_labels).reset_index(drop=True)

# Label statistics
positive_labels = labeled_data['markdown_next_30d'].sum()
total_labels = len(labeled_data)
label_rate = (positive_labels / total_labels) * 100

print(f"✅ Created prediction labels (sample):")
print(f"   Positive labels (markdown in 30d): {positive_labels:,}")
print(f"   Total labels: {total_labels:,}")
print(f"   Positive rate: {label_rate:.2f}%")

print(f"\n5. SAMPLE MARKDOWN ANALYSIS")
print("-" * 40)

# Show examples of detected markdowns
markdown_examples = labeled_data[labeled_data['is_markdown'] == True].head(10)
print(f"Sample markdown events:")
for _, row in markdown_examples.iterrows():
    print(f"   Article {row['article_id']:>8} on {row['t_dat'].strftime('%Y-%m-%d')}: "
          f"${row['avg_price']:>6.2f} (was ${row['rolling_median_price']:>6.2f}, "
          f"{row['price_drop_pct']*100:>5.1f}% drop)")

print(f"\n6. SAVING PROCESSED DATA")
print("-" * 40)

# Save labeled dataset
output_path = processed_dir / "markdown_labels_sample.csv"
labeled_data.to_csv(output_path, index=False)
print(f"✅ Saved labeled dataset: {output_path}")
print(f"   Records: {len(labeled_data):,}")
print(f"   Features: {list(labeled_data.columns)}")

# Save summary statistics
summary_stats = {
    'total_transactions': len(transactions),
    'unique_articles': transactions['article_id'].nunique(),
    'date_range_start': transactions['t_dat'].min().strftime('%Y-%m-%d'),
    'date_range_end': transactions['t_dat'].max().strftime('%Y-%m-%d'),
    'markdown_threshold': MARKDOWN_THRESHOLD,
    'rolling_window_days': ROLLING_WINDOW_DAYS,
    'prediction_horizon_days': PREDICTION_HORIZON_DAYS,
    'markdown_events_detected': int(markdown_events),
    'markdown_rate_pct': round(markdown_rate, 2),
    'positive_label_rate_pct': round(label_rate, 2)
}

import json
stats_path = processed_dir / "markdown_label_stats.json"
with open(stats_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"✅ Saved summary statistics: {stats_path}")

print(f"\n7. NEXT STEPS SUMMARY")
print("-" * 40)
print(f"✅ Markdown detection pipeline complete!")
print(f"✅ Labels created for {PREDICTION_HORIZON_DAYS}-day prediction horizon")
print(f"✅ Ready for feature engineering and model training")
print(f"\nFiles created:")
print(f"   📁 {output_path}")
print(f"   📁 {stats_path}")
print(f"\nNext: Feature engineering (product, pricing, demand signals)")

print("=" * 50)
print("MARKDOWN LABEL CREATION COMPLETE! ✅")
print("=" * 50)
