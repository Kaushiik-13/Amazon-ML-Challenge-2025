
# ML Challenge 2025: Smart Product Pricing Solution

*Team Name:* Basement
*Team Members:* Kaushiik A, Kavin Mohan Kumar,Suhail S,Nikitha M
*Submission Date:* 13-10-2025

---

## 1. Executive Summary

We implemented a **LightGBM regression model** to predict product prices using catalog text. Our solution combines **TF-IDF embeddings**, **numeric features** derived from text, and **target-encoded brand/IPQ features**, achieving a validation SMAPE of 0.49. This hybrid approach allows the model to capture both textual semantics and structured patterns in the dataset.

---

## 2. Methodology Overview

### Problem Analysis

The task involves predicting prices for a **heterogeneous product catalog** with variable textual descriptions.

**Insights from Exploratory Analysis (EDA & Feature Inspection):**

1. **Catalog content length matters:**

   * Word count and character count correlate moderately with price. Longer descriptions often indicate higher-priced products.
2. **Numerical indicators in text (IPQ)**:

   * Many catalog entries contain numeric values (e.g., “500ml”, “250g”).
   * Extracting the first numeric value as **IPQ** allows the model to infer approximate product size or quantity, which affects pricing.
3. **Brand effect:**

   * The first word of catalog text is usually the brand.
   * Using target encoding (mean price per brand) allows the model to leverage brand reputation and historical pricing.
4. **Textual embeddings via TF-IDF:**

   * Captures specific product descriptors, adjectives, and qualifiers (e.g., “premium”, “organic”) which influence price.
5. **Log transformation of price:**

   * Stabilizes variance, reduces effect of extreme values, and improves model convergence.

### Solution Strategy

* **Hybrid approach:** Combines **text embeddings (TF-IDF)**, **engineered numeric features**, and **target-encoded features** in a LightGBM regression.
* **Modeling insights:**

  * LightGBM handles high-dimensional sparse inputs efficiently.
  * Ensemble tree-based methods capture non-linear interactions between text, numeric, and target-encoded features.

*Approach Type:* Hybrid (TF-IDF + Numeric + Target Encoding)
*Core Innovation:* Integrating target-encoded brand and IPQ features with TF-IDF embeddings to model heterogeneous product data effectively.

---

## 3. Model Architecture

### Architecture Overview

```
Catalog Content (Text) ------------------------|
                                               |
Numeric Features (Word count, Char count, IPQ) |--> Feature Concatenation --> LightGBM --> Log-Price Prediction
                                               |
Target-Encoded Features (Brand_mean, IPQ_mean) |
```

### Model Components

**Text Processing Pipeline:**

* Fill missing catalog content with empty strings.
* TF-IDF vectorization:

  * 1-2 grams, max 50,000 features, stop words removed.
* Converts textual descriptions into sparse numeric embeddings.

**Numeric Features Pipeline:**

* Extracted from catalog content:

  * Word count
  * Character count
  * IPQ (first numeric occurrence; default 1.0)
* Helps model understand the structural and size aspects of products.

**Target Encoding Pipeline:**

* Brand mean price (`brand_mean`)
* IPQ mean price (`ipq_mean`)
* Provides prior knowledge of typical pricing based on brand and size.

**LightGBM Configuration:**

* Learning rate: 0.025
* num_leaves: 127
* feature_fraction: 0.8
* bagging_fraction: 0.8, bagging_freq: 5
* min_data_in_leaf: 20
* Objective: regression, metric: RMSE
* 800 boosting rounds

**Pipeline Flow:**

1. **Build features:** TF-IDF + numeric + target-encoded
2. **Train-validation split:** 85% train, 15% validation
3. **Log-transform prices:** `y_train = log1p(price)`
4. **Train LightGBM** on training features
5. **Validate:** predict, inverse log (`expm1`), calculate SMAPE
6. **Retrain on full dataset** for final model deployment

---

## 4. Feature Analysis

| Feature Type              | Details/Extraction Method                               | Importance Insights                         |
| ------------------------- | ------------------------------------------------------- | ------------------------------------------- |
| Text (TF-IDF)             | 1-2 grams, max 50k features                             | Captures descriptive words affecting price  |
| Word Count                | Number of words in `catalog_content`                    | Moderate positive correlation with price    |
| Character Count           | Number of characters in `catalog_content`               | Helps capture verbosity & complexity        |
| IPQ (Extracted Number)    | First numeric occurrence in catalog text; default 1.0   | Encodes product size/quantity               |
| Brand Mean (`brand_mean`) | Mean price of products with same brand in training data | High predictive power, encodes brand effect |
| IPQ Mean (`ipq_mean`)     | Mean price of products with same IPQ in training data   | Helps model correlate size with price       |

---

## 5. Model Performance

### Validation Results

* Validation SMAPE: **0.4995**
* Log-transformed RMSE (training/validation): ~0.35
* MAE (log price): ~0.28

**Observations:**

* TF-IDF captures product-specific descriptors but may be sparse for rare products.
* Numeric and target-encoded features help reduce bias in extreme price ranges.
* Overall SMAPE is high due to dataset diversity; further improvements possible via:

  * Fine-tuning LightGBM parameters
  * Increasing TF-IDF features or using embeddings (e.g., BERT)
  * Incorporating product images or multimodal features

---

## 6. Conclusion

The hybrid LightGBM model leverages **text, numeric, and target-encoded features** to predict heterogeneous product prices.

* **Strengths:** Robust for diverse catalog content, interpretable feature importance, easy to retrain.
* **Limitations:** SMAPE of 0.49 indicates high variability in some product categories.
* **Future Work:** Integrate richer embeddings (language models), expand numeric features, explore ensemble approaches to further reduce SMAPE.

---

