import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

def extract_numeric_features(df):
    """Word count, char count, IPQ"""
    word_count = df['catalog_content'].apply(lambda x: len(str(x).split())).values.reshape(-1,1)
    char_count = df['catalog_content'].apply(lambda x: len(str(x))).values.reshape(-1,1)
    ipq = df['catalog_content'].apply(lambda x: float(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 1.0)
    ipq = ipq.values.reshape(-1,1)
    return np.hstack([word_count, char_count, ipq])

def extract_target_encoding_features(df, full_df):
    """Brand and IPQ mean price encoding"""
    df = df.copy()
    full_df = full_df.copy()
    
    # brand
    df['brand'] = df['catalog_content'].apply(lambda x: str(x).split()[0].lower() if len(str(x).split())>0 else 'unknown')
    full_df['brand'] = full_df['catalog_content'].apply(lambda x: str(x).split()[0].lower() if len(str(x).split())>0 else 'unknown')

    # IPQ
    df['ipq'] = df['catalog_content'].apply(lambda x: float(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 1.0)
    full_df['ipq'] = full_df['catalog_content'].apply(lambda x: float(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 1.0)

    # mean price maps
    brand_mean_map = full_df.groupby('brand')['price'].mean().to_dict()
    ipq_mean_map = full_df.groupby('ipq')['price'].mean().to_dict()

    df['brand_mean'] = df['brand'].map(brand_mean_map).fillna(full_df['price'].mean())
    df['ipq_mean'] = df['ipq'].map(ipq_mean_map).fillna(full_df['price'].mean())

    return df[['brand_mean', 'ipq_mean']].values

def build_features(df, full_df=None, tfidf=None, fit_tfidf=True, max_features=50000, ngram_range=(1,2)):
    """Combine TF-IDF + numeric + target-encoded features"""
    text_data = df['catalog_content'].fillna('').astype(str)

    if fit_tfidf:
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words='english')
        X_text = tfidf.fit_transform(text_data)
    else:
        X_text = tfidf.transform(text_data)

    X_num = extract_numeric_features(df)
    X_num_sparse = csr_matrix(X_num)

    X_target_sparse = csr_matrix(extract_target_encoding_features(df, full_df)) if full_df is not None else csr_matrix(np.zeros((df.shape[0],2)))

    X = hstack([X_text, X_num_sparse, X_target_sparse])
    return X, tfidf
