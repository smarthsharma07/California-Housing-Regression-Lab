# California Housing Price Prediction

A structured machine learning regression project focused on predicting California housing prices using classical linear models, thoughtful preprocessing, and feature engineering.

The goal of this project was not only to train models, but to understand how:
- data distributions affect model behavior,
- preprocessing influences optimization,
- feature engineering improves signal quality,
- and where linear regression begins to fail on nonlinear real-world data.

---

# Project Objective

The project aims to build a complete end-to-end regression workflow using the California Housing dataset.

Instead of focusing only on model accuracy, the project emphasizes:
- Exploratory Data Analysis (EDA)
- preprocessing strategy design
- feature engineering
- train/validation/test discipline
- model comparison
- optimization behavior
- and practical ML reasoning

---

# Dataset

The project uses the California Housing dataset containing:
- demographic information
- housing statistics
- income data
- geographical coordinates
- ocean proximity categories

### Target Variable

`median_house_value`

---

# Project Workflow

## 1. Exploratory Data Analysis (EDA)

Performed:
- histogram analysis
- boxplots
- scatter plots
- correlation heatmaps
- missing value analysis

Key observations:
- several numerical features showed strong right skewness
- multiple outliers were present
- `median_income` showed the strongest correlation with house prices
- geographic features exhibited nonlinear clustered behavior
- housing prices appeared capped near 500k

---

## 2. Data Splitting

Used proper:
- Train Set
- Validation Set
- Test Set

Split Ratio:
- 70% Train
- 15% Validation
- 15% Test

This ensured:
- unbiased evaluation
- cleaner experimentation
- and reduced risk of data leakage

All preprocessing steps were fitted only on training data.

---

## 3. Missing Value Handling

Feature with missing values:
- `total_bedrooms`

Applied:
- Median Imputation

Reason:
- more robust against skewed distributions and outliers compared to mean imputation.

---

## 4. Feature Engineering

Created ratio-based features:
- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

These features provided more meaningful representations than raw count-based totals.

Observations:
- reduced redundancy among highly correlated features
- improved interpretability
- captured district-level housing behavior more effectively

---

## 5. Categorical Encoding

Categorical feature:
- `ocean_proximity`

Applied:
- One-Hot Encoding

Used:
```python
drop='first'
```

to avoid multicollinearity in linear models.

---

## 6. Preprocessing Strategy

Different preprocessing techniques were applied based on feature distributions instead of using a single scaling strategy everywhere.

### Log Transformations

Applied to heavily skewed features to:
- reduce extreme tails
- stabilize variance
- improve feature distributions

---

### RobustScaler

Used on outlier-heavy features because it relies on:
- median
- interquartile range (IQR)

making it more resistant to extreme values.

---

### StandardScaler

Used on relatively stable distributions to improve optimization efficiency.

---

# Models Trained

## Linear Regression

Baseline model used to establish reference performance.

### Validation Metrics
- MAE ≈ 49.8k
- RMSE ≈ 70k
- R² ≈ 0.623

---

## Ridge Regression

Applied L2 regularization using `RidgeCV`.

Observation:
- negligible improvement over baseline regression
- suggested coefficients were already relatively stable

---

## Lasso Regression

Applied L1 regularization using `LassoCV`.

Observation:
- selected very small alpha
- most engineered features already carried useful signal

---

## SGDRegressor

Explored gradient descent optimization through:
- learning rate tuning
- regularization
- epochs
- learning schedules

Used:
- `RandomizedSearchCV`

Observation:
- converged close to Linear Regression performance
- optimization was not the main limitation

---

# Final Test Results

| Metric | Value |
|---|---|
| MAE | ~48.3k |
| RMSE | ~69.8k |
| R² Score | ~0.631 |

The close agreement between validation and test metrics indicated:
- stable generalization
- minimal leakage
- and consistent preprocessing behavior

---

# Major Learnings

This project reinforced several important machine learning concepts:

- preprocessing quality strongly affects model performance
- feature engineering can outperform model complexity
- regularization is not universally beneficial
- optimization improvements cannot overcome model capacity limitations
- structured experimentation matters more than blindly switching algorithms

The project also improved practical intuition regarding:
- skewed distributions
- outlier handling
- scaling strategies
- regression workflows
- and feature representation

---

# Important Insight

One of the key observations from this project was:

> Even strong optimization techniques cannot fully overcome the limitations of linear models when the underlying dataset contains nonlinear relationships.

The California housing dataset contains:
- spatial clustering
- nonlinear geographic behavior
- complex feature interactions
- and non-uniform distributions

Although preprocessing and optimization improved convergence and stability, linear models eventually reached a performance ceiling because the hypothesis itself remained linear.

This highlighted an important ML principle:

> Better optimization improves learning efficiency, but model architecture determines what patterns can actually be learned.

---

# Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Future Improvements

Possible next steps:
- Random Forest Regressor
- Gradient Boosting
- XGBoost
- Spatial feature engineering
- Cross-validation experiments
- Ensemble approaches

Tree-based models may better capture nonlinear geographic relationships present in housing datasets.

---

# Repository Structure

```text
├── house-prices-predictor-linear-regression.ipynb
├── README.md
└── requirements.txt
```

---

# Running the Project

## Clone Repository

```bash
git clone <your-repository-link>
cd <repository-name>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Launch Notebook

```bash
jupyter notebook
```

---

# Final Note

This project was built as part of a deeper effort to understand:
- machine learning workflows,
- preprocessing pipelines,
- feature engineering,
- optimization behavior,
- and practical regression modeling beyond tutorial-level implementation.
