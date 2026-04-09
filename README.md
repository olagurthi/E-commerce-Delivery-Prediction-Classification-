# 📦 E-Commerce Delivery Status Prediction
### Machine Learning Classification Project — Phase 1: EDA & Preprocessing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle" />
  <img src="https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen" />
  <img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey" />
</p>

---

## 📋 Project Overview

Can we predict whether an e-commerce shipment will be **delivered on time, delayed, or returned** — before it even leaves the warehouse?

This project applies machine learning to 50,000 synthetic e-commerce orders to classify delivery outcomes. Accurate prediction enables logistics teams to intervene early, reduce delays, and improve customer satisfaction.

| Item | Details |
|------|---------|
| **Task** | Multi-class Classification |
| **Target** | `Delivery_Status` — Delivered / Delayed / Returned |
| **Dataset** | E-Commerce Shipping & Delivery Performance (50K Records) |
| **Source** | [Kaggle](https://www.kaggle.com/) |
| **Models (Phase 2)** | Logistic Regression · SVM · XGBoost |

---

## 🗂️ Repository Structure

```
📦 ecommerce-delivery-prediction/
├── 📓 phase1_eda_preprocessing.ipynb   ← Phase 1: EDA & Preprocessing (this file)
├── 📓 phase2_classification.ipynb       ← Phase 2: Model Training & Evaluation
├── 📄 README.md                         ← You are here
├── 📊 data/
│   └── E-Commerce_Order_Fulfillment_Dataset_50K.csv
├── 📁 outputs/
│   ├── X_train.csv                      ← Preprocessed training features
│   ├── X_test.csv                       ← Preprocessed test features
│   ├── y_train.csv                      ← Training labels
│   ├── y_test.csv                       ← Test labels
│   └── feature_columns.json            ← Final feature list
└── 📁 figures/
    ├── fig1_target_distribution.png
    ├── fig2_shipping_mode.png
    ├── fig3_region_analysis.png
    └── ...
```

---

## 📊 Dataset

| Column | Type | Description |
|--------|------|-------------|
| `Order_ID` | String | Unique order identifier |
| `Customer_Region` | Categorical | North / South / East / West / Central |
| `Product_Category` | Categorical | Electronics, Fashion, Home, Grocery, Beauty, Sports |
| `Order_Date` | Date | When the order was placed |
| `Ship_Date` | Date | When the order was dispatched |
| `Delivery_Date` | Date | When the order arrived |
| `Shipping_Mode` | Categorical | Standard / Express / Same Day |
| `Shipping_Cost` | Numeric | Cost of shipping (USD) |
| `Delivery_Status` | **Target** | **Delivered / Delayed / Returned** |
| `Delivery_Days` | Numeric | Days from shipment to delivery |

**Size:** 50,000 rows · 10 columns · 0 missing values · 0 duplicates

---

## 🔬 Phase 1 — EDA & Preprocessing

### What's Inside the Notebook

The notebook is organized into 4 clearly commented sections:

#### 1️⃣ Problem Definition & Dataset Understanding
- Business problem formulation
- ML task specification
- Initial data inspection (shape, dtypes, nulls, duplicates)
- Statistical summaries

#### 2️⃣ Exploratory Data Analysis (11 visualizations)

| Figure | Description |
|--------|-------------|
| Fig 1 | Target class distribution (bar + pie) |
| Fig 2 | Delivery status breakdown by Shipping Mode |
| Fig 3 | Delivery performance by Customer Region |
| Fig 4 | Delivery rates by Product Category |
| Fig 5 | Delivery Days — histogram & boxplot by status |
| Fig 6 | Shipping Cost — violin plot & histogram |
| Fig 7 | Shipping Cost by Shipping Mode (boxplot) |
| Fig 8 | Monthly order trends 2022–2025 |
| Fig 9 | Correlation heatmap |
| Fig 10 | Outlier detection (IQR boxplots) |
| Fig 11 | Scatter: Shipping Cost vs Delivery Days |

#### 3️⃣ Preprocessing
- Date parsing (`Order_Date`, `Ship_Date`, `Delivery_Date`)
- Drop non-predictive ID column
- Label Encoding for categorical features
- Target variable encoding

#### 4️⃣ Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `processing_time` | `Ship_Date − Order_Date` | Warehouse fulfilment speed |
| `order_month` | `Order_Date.month` | Seasonal demand patterns |
| `order_dayofweek` | `Order_Date.dayofweek` | Weekend shipping lag |
| `order_year` | `Order_Date.year` | Year-over-year trend |
| `is_express` | `Shipping_Mode == Express` | Binary priority flag |
| `is_same_day` | `Shipping_Mode == Same Day` | Binary rush flag |
| `cost_per_day` | `Shipping_Cost / Delivery_Days` | Efficiency proxy |
| OHE features | `get_dummies(Region, Category)` | No ordinal assumption |

**Final feature count: ~23 features**  
**Train/Test split: 80% / 20% (stratified)**  
**Scaler: StandardScaler**

---

## 🔑 Key EDA Findings

> 1. **Class imbalance** — 80.1% Delivered · 15.0% Delayed · 4.9% Returned → use F1-macro
> 2. **Delivery_Days** is the strongest single separator between Delivered and Delayed orders
> 3. **Processing time** is meaningfully higher for delayed orders
> 4. **Cost per day** is higher for successfully delivered orders
> 5. **Regional differences** are minor — all 5 regions roughly equally represented
> 6. **No critical outliers** — values fall within realistic logistics ranges
> 7. **No missing data** — dataset is complete and ready for modeling

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### For Phase 2 (coming soon)
```bash
pip install xgboost scikit-learn
```

### Run the notebook

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-delivery-prediction.git
cd ecommerce-delivery-prediction
jupyter notebook phase1_eda_preprocessing.ipynb
```

> **Note:** Place the dataset CSV in the same directory as the notebook, or update the `pd.read_csv()` path in Cell 2.

---

## 🚀 Phase 2 — Coming Next

Phase 2 will implement and compare three classifiers using the same preprocessed data:

| Model | Role | Why |
|-------|------|-----|
| **Logistic Regression** | Baseline | Interpretable, fast, linear boundary |
| **SVM** | Core model | Strong on high-dimensional scaled data |
| **XGBoost** | Advanced model | Handles class imbalance, non-linear patterns |

Evaluation will include: Accuracy · F1-macro · Precision · Recall · ROC-AUC · Confusion Matrix

---

## 👥 Team

| Name | Role |
|------|------|
| Student 1 | EDA & Visualizations |
| Student 2 | Preprocessing & Feature Engineering |
| Student 3 | Documentation & Presentation |

---

## 📄 License

Dataset licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).  
Code in this repository is available under the MIT License.

---

## 🤖 AI Usage Statement

This project used AI assistance (Claude by Anthropic) for:
- Structuring the Jupyter notebook layout and comments
- Suggesting feature engineering ideas
- Generating initial visualization code scaffolding

All analysis, interpretation, and conclusions were reviewed and validated by the student team.

---

<p align="center">
  <em>Built as part of a Machine Learning course project · 2025</em>
</p>
