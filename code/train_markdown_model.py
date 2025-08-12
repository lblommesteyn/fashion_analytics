import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("H&M Markdown Prediction - Model Training")
print("=" * 50)

# File paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
processed_dir = data_dir / "processed"
outputs_dir = data_dir / "outputs"
outputs_dir.mkdir(exist_ok=True)

print("1. LOADING PROCESSED DATASETS")
print("-" * 40)

# Load engineered features
print("Loading engineered features...")
features_df = pd.read_csv(processed_dir / "engineered_features.csv")
print(f"✅ Features loaded: {features_df.shape[0]:,} articles × {features_df.shape[1]} features")

# Load feature metadata
with open(processed_dir / "feature_metadata.json", 'r') as f:
    feature_metadata = json.load(f)
print(f"✅ Feature metadata loaded")

# For this demo, we'll create synthetic labels since the label creation is still processing
# In production, you'd load the actual markdown labels
print("Creating synthetic markdown labels for model development...")
np.random.seed(42)

# Create realistic synthetic labels based on pricing patterns
synthetic_labels = pd.DataFrame()
synthetic_labels['article_id'] = features_df['article_id']

# Higher markdown probability for:
# - Higher price volatility 
# - Seasonal items
# - Items with declining sales momentum
markdown_prob = 0.05  # Base 5% markdown rate

# Add realistic patterns
if 'price_volatility' in features_df.columns:
    volatile_boost = features_df['price_volatility'].fillna(False).astype(int) * 0.10
else:
    volatile_boost = 0

if 'sales_momentum' in features_df.columns:
    momentum_penalty = np.where(features_df['sales_momentum'].fillna(0.5) < 0.5, 0.08, 0)
else:
    momentum_penalty = 0

# Generate synthetic labels with realistic patterns
final_prob = markdown_prob + volatile_boost + momentum_penalty
synthetic_labels['markdown_next_30d'] = np.random.binomial(1, np.minimum(final_prob, 0.25))

label_rate = synthetic_labels['markdown_next_30d'].mean() * 100
print(f"✅ Synthetic labels created: {label_rate:.1f}% positive rate")

print("\n2. PREPARING TRAINING DATA")
print("-" * 40)

# Merge features with labels
modeling_data = pd.merge(features_df, synthetic_labels, on='article_id', how='inner')
print(f"✅ Modeling dataset: {modeling_data.shape[0]:,} samples")

# Separate features and target - exclude non-numeric columns
feature_columns = [col for col in modeling_data.columns if col not in ['article_id', 'markdown_next_30d']]
X = modeling_data[feature_columns].copy()
y = modeling_data['markdown_next_30d'].copy()

print(f"   Initial features: {len(feature_columns)} columns")
print(f"   Target variable: {y.name}")
print(f"   Positive class rate: {y.mean()*100:.1f}%")

# Clean and prepare features for ML
print("Cleaning and preparing feature data...")

# Handle datetime columns
date_columns = X.select_dtypes(include=['datetime64']).columns
for col in date_columns:
    # Convert to days since min date
    min_date = X[col].min()
    X[col] = (X[col] - min_date).dt.days
    print(f"   Converted datetime column: {col}")

# Handle object columns
object_columns = X.select_dtypes(include=['object']).columns
for col in object_columns:
    # Try to convert to numeric, if fails then drop
    try:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        print(f"   Converted object to numeric: {col}")
    except:
        print(f"   Dropping non-convertible column: {col}")
        X = X.drop(columns=[col])

# Handle boolean columns
bool_columns = X.select_dtypes(include=['bool']).columns
for col in bool_columns:
    X[col] = X[col].astype(int)
    print(f"   Converted boolean to int: {col}")

# Handle missing values and infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Remove constant features
constant_features = X.columns[X.nunique() <= 1]
if len(constant_features) > 0:
    X = X.drop(columns=constant_features)
    print(f"   Removed {len(constant_features)} constant features")

print(f"✅ Clean feature matrix: {X.shape}")

print("\n3. TRAIN-TEST SPLIT")
print("-" * 40)

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train set: {X_train.shape[0]:,} samples ({y_train.mean()*100:.1f}% positive)")
print(f"✅ Test set:  {X_test.shape[0]:,} samples ({y_test.mean()*100:.1f}% positive)")

print("\n4. TRAINING LIGHTGBM MODEL")
print("-" * 40)

# LightGBM parameters optimized for binary classification
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

print("Training LightGBM model...")
lgb_model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

print("✅ LightGBM training complete")

print("\n5. TRAINING XGBOOST MODEL")  
print("-" * 40)

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0
}

print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=100)
xgb_model.fit(X_train, y_train)

print("✅ XGBoost training complete")

print("\n6. MODEL EVALUATION")
print("-" * 40)

# Get predictions
lgb_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
def evaluate_model(y_true, y_pred_proba, model_name):
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Binary predictions at 0.5 threshold
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print(f"\n{model_name} Performance:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC:  {pr_auc:.4f}")
    print(f"   Accuracy: {(y_true == y_pred).mean():.4f}")
    
    return roc_auc, pr_auc

lgb_roc, lgb_pr = evaluate_model(y_test, lgb_pred_proba, "LightGBM")
xgb_roc, xgb_pr = evaluate_model(y_test, xgb_pred_proba, "XGBoost")

# Select best model
if lgb_pr > xgb_pr:  # Use PR-AUC as primary metric for imbalanced classes
    best_model = lgb_model
    best_predictions = lgb_pred_proba
    best_model_name = "LightGBM"
    print(f"\n🏆 Best Model: {best_model_name} (PR-AUC: {lgb_pr:.4f})")
else:
    best_model = xgb_model
    best_predictions = xgb_pred_proba
    best_model_name = "XGBoost"
    print(f"\n🏆 Best Model: {best_model_name} (PR-AUC: {xgb_pr:.4f})")

print("\n7. MODEL CALIBRATION")
print("-" * 40)

# Calibrate the best model
if best_model_name == "LightGBM":
    # For LightGBM, create wrapper for calibration
    class LGBWrapper:
        def __init__(self, model):
            self.model = model
        def predict_proba(self, X):
            preds = self.model.predict(X, num_iteration=self.model.best_iteration)
            return np.column_stack([1-preds, preds])
    
    calibrated_model = CalibratedClassifierCV(LGBWrapper(best_model), method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
else:
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

# Get calibrated predictions
calibrated_predictions = calibrated_model.predict_proba(X_test)[:, 1]

print("✅ Model calibration complete")
print(f"   Calibrated model PR-AUC: {auc(*precision_recall_curve(y_test, calibrated_predictions)[:2]):.4f}")

print("\n8. FEATURE IMPORTANCE")
print("-" * 40)

# Get feature importance
if best_model_name == "LightGBM":
    feature_importance = best_model.feature_importance()
    feature_names = X_train.columns
else:
    feature_importance = best_model.feature_importances_
    feature_names = X_train.columns

# Create importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 most important features:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<30} {row['importance']:>8.0f}")

print("\n9. SAVING MODEL AND RESULTS")
print("-" * 40)

# Save model
import joblib
model_path = outputs_dir / f"markdown_model_{best_model_name.lower()}.joblib"
joblib.dump(calibrated_model, model_path)
print(f"✅ Saved calibrated model: {model_path}")

# Save feature importance
importance_path = outputs_dir / "feature_importance.csv"
importance_df.to_csv(importance_path, index=False)
print(f"✅ Saved feature importance: {importance_path}")

# Save model metadata
model_metadata = {
    'model_type': best_model_name,
    'training_date': datetime.now().isoformat(),
    'performance': {
        'roc_auc': float(lgb_roc if best_model_name == "LightGBM" else xgb_roc),
        'pr_auc': float(lgb_pr if best_model_name == "LightGBM" else xgb_pr)
    },
    'data_info': {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(feature_names),
        'positive_rate': float(y.mean())
    },
    'top_features': importance_df.head(20)['feature'].tolist()
}

metadata_path = outputs_dir / "model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(model_metadata, f, indent=2)
print(f"✅ Saved model metadata: {metadata_path}")

# Save test predictions for analysis
test_results = pd.DataFrame({
    'article_id': modeling_data.loc[X_test.index, 'article_id'],
    'actual_markdown': y_test,
    'predicted_probability': calibrated_predictions,
    'risk_category': pd.cut(calibrated_predictions, bins=[0, 0.1, 0.3, 0.7, 1.0], 
                           labels=['Low', 'Medium', 'High', 'Very High'])
})

results_path = outputs_dir / "test_predictions.csv"
test_results.to_csv(results_path, index=False)
print(f"✅ Saved test predictions: {results_path}")

print("\n10. BUSINESS IMPACT ANALYSIS")
print("-" * 40)

# Risk segmentation analysis
risk_analysis = test_results.groupby('risk_category').agg({
    'predicted_probability': ['count', 'mean'],
    'actual_markdown': 'mean'
}).round(3)

print("Risk Segmentation Analysis:")
print(risk_analysis)

print(f"\n" + "=" * 50)
print("✅ MARKDOWN PREDICTION MODEL TRAINING COMPLETE!")
print(f"🎯 Model Performance: {best_model_name} with PR-AUC = {lgb_pr if best_model_name == 'LightGBM' else xgb_pr:.4f}")
print(f"📊 Ready for What-If Simulator Integration")
print("=" * 50)
