# src/predict.py
import os
import joblib
import numpy as np
import pandas as pd
from features import build_features
import joblib
import lightgbm as lgb

MODEL_DIR = "models"

def main():
    test = pd.read_csv("dataset/test.csv")
    test['catalog_content'] = test['catalog_content'].fillna('').astype(str)
    model = joblib.load("models/lgbm_log1p.pkl")
    tfidf = joblib.load("models/tfidf.joblib")



    X_test, _ = build_features(test, tfidf=tfidf, fit_tfidf=False)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    # ensure positive floats
    preds = np.clip(preds, 0.01, None)

    out = pd.DataFrame({
        "sample_id": test['sample_id'],
        "price": preds.astype(float)
    })

    # Ensure same number of rows as test
    assert len(out) == len(test), "Row count mismatch with test.csv"

    out.to_csv("test_out.csv", index=False)
    print("Saved test_out.csv (rows: {})".format(len(out)))

if __name__ == "__main__":
    main()
