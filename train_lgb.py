import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from features import build_features

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom == 0
    res = np.zeros_like(denom)
    res[~mask] = np.abs(y_true[~mask] - y_pred[~mask]) / denom[~mask]
    return np.mean(res)

def main():
    df = pd.read_csv("../dataset/train.csv")
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)
    df = df[df['price'].notna()].copy()
    df['price'] = df['price'].clip(lower=0.0)

    # Train/validation split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    print("Building features (train)...")
    X_train, tfidf = build_features(train_df, fit_tfidf=True)
    X_val, _ = build_features(val_df, tfidf=tfidf, fit_tfidf=False)

    y_train = np.log1p(train_df['price'].values)
    y_val = val_df['price'].values

    # LightGBM dataset
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=np.log1p(y_val), reference=lgb_train)

    # Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.025,   # lower learning rate
        'num_leaves': 127,        # higher complexity
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 20
    }


    print("Training LightGBM on train/val split...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=800
    )

    # Predict on validation
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    score = smape(y_val, val_preds)
    print(f"Validation SMAPE: {score:.6f}")

    # Retrain on full dataset
    print("Retraining on full dataset...")
    X_full, tfidf_full = build_features(df, fit_tfidf=True)
    y_full = np.log1p(df['price'].values)
    full_train = lgb.Dataset(X_full, label=y_full)
    model = lgb.train(params, full_train, num_boost_round=800)

    # Save model and TF-IDF
    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_log1p.pkl"))
    joblib.dump(tfidf_full, os.path.join(MODEL_DIR, "tfidf.joblib"))
    print("Saved LightGBM model to", MODEL_DIR)

    with open(os.path.join(MODEL_DIR, "metrics_lgb.txt"), "w") as f:
        f.write(f"validation_smape={score:.6f}\n")

if __name__ == "__main__":
    main()
