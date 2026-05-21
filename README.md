# California Housing Regression Lab

A comprehensive machine learning regression project focused on predicting California housing prices using multiple regression approaches including:

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- SGDRegressor
- and upcoming tree-based & kernel-based models.

The goal of this project is not only to improve prediction accuracy, but also to deeply understand:
- preprocessing strategies,
- feature engineering,
- nonlinear learning,
- optimization behavior,
- regularization,
- and practical machine learning experimentation workflows.

---

# Project Objective

This project explores an end-to-end regression workflow using the California Housing dataset.

The focus is placed on:
- Exploratory Data Analysis (EDA)
- preprocessing strategy design
- feature engineering
- train/validation/test discipline
- model comparison
- optimization behavior
- regularization
- and understanding how different regression models behave on real-world nonlinear datasets.

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

These observations strongly suggested that simple linear models alone may not fully capture the underlying data relationships.

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

These engineered features provided more meaningful representations than raw count-based totals.

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

to avoid multicollinearity in linear-family models.

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

# Models Explored

## 1. Linear Regression

Used as the baseline regression model.

Observations:
- performed reasonably on structured numerical features
- struggled with nonlinear geographic relationships
- showed limitations in capturing spatial clustering behavior

### Validation Performance
- R² ≈ 0.62

---

## 2. Polynomial Regression

To capture nonlinear feature interactions, Polynomial Features (Degree 2) were introduced.

This allowed the model to learn:
- interaction terms
- quadratic relationships
- nonlinear feature combinations

Different polynomial degrees were tested:
- Degree 1 behaved similarly to baseline regression
- Degree 2 significantly improved performance
- Degree 3 caused severe overfitting and instability

Observations:
- captured nonlinear geographic and income-based behavior effectively
- significantly improved validation performance
- demonstrated the importance of nonlinear feature interactions

### Best Validation Performance
- R² ≈ 0.663

---

## 3. Polynomial + Ridge Regression

Applied Ridge Regression (L2 Regularization) on top of degree-2 polynomial features.

Purpose:
- stabilize correlated polynomial features
- reduce coefficient explosion
- improve generalization

Multiple alpha values were tested:
- 0.001
- 0.01
- 0.1
- 1
- 10
- 100

Observations:
- produced the best overall validation performance
- handled multicollinearity effectively
- improved numerical stability

### Best Validation Performance
- R² ≈ 0.665

### Final Test Performance
- R² ≈ 0.646

This became the best-performing classical regression model in the project.

---

## 4. Polynomial + Lasso Regression

Applied Lasso Regression (L1 Regularization) on polynomial features.

Purpose:
- feature selection
- coefficient sparsity
- regularization under correlated interactions

Observations:
- performed similarly to Ridge Regression
- convergence warnings appeared for very small alpha values
- struggled more with correlated polynomial terms

### Best Validation Performance
- R² ≈ 0.664

This experiment showed that aggressive sparsity constraints are not always ideal in highly correlated polynomial feature spaces.

---

## 5. Polynomial + SGDRegressor

Explored iterative optimization using SGDRegressor.

Experiments included:
- learning rate tuning
- regularization tuning
- target scaling
- convergence stabilization

Initial runs suffered from:
- exploding gradients
- optimizer divergence
- extremely unstable predictions

To stabilize optimization:
- target scaling was introduced
- smaller learning rates were tested
- regularization strength was tuned

Observations:
- optimization became stable after target scaling
- SGD remained highly sensitive to hyperparameters
- optimization quality alone could not outperform Ridge Regression

### Best Validation Performance
- R² ≈ 0.634

This experiment highlighted the importance of optimization dynamics in gradient-based learning systems.

---

# Final Selected Model

Best Overall Model:
- Polynomial Features (Degree 2)
- Ridge Regression (alpha = 1)

---

# Final Test Results

| Metric | Value |
|---|---|
| MAE | ~45.5k |
| RMSE | ~67.9k |
| R² Score | ~0.646 |

The close agreement between validation and test metrics indicated:
- good generalization
- minimal leakage
- stable preprocessing
- and successful nonlinear feature learning.

---

# Major Insights From The Project

## Nonlinearity Matters

EDA revealed that:
- longitude and latitude showed clustered nonlinear relationships
- housing prices were geographically dependent
- simple linear models could not fully capture spatial behavior

Polynomial feature expansion significantly improved learning capacity.

---

## Feature Engineering Improved Representation

Ratio-based engineered features:
- reduced redundancy
- improved interpretability
- and captured district-level housing characteristics more effectively.

---

## More Complexity Is Not Always Better

Degree-3 polynomial expansion dramatically increased feature count and caused:
- overfitting
- instability
- poor generalization

This reinforced the importance of balancing model complexity and generalization.

---

## Regularization Stabilizes Complex Feature Spaces

Polynomial expansion introduced strong feature correlation.

- Ridge Regression stabilized coefficients effectively.
- Lasso Regression struggled more with correlated polynomial terms.

This demonstrated why regularization becomes increasingly important as feature complexity grows.

---

## Optimization ≠ Model Capacity

SGD experiments showed that:
- optimization can fail even when the feature space is correct
- learning rate selection is critical
- stable optimization does not guarantee best performance

This highlighted the difference between:
- optimization quality
- and representational capacity.

---

## Proper Experimental Discipline Matters

The dataset was split before preprocessing to avoid leakage:
- Train Set → fitting
- Validation Set → tuning
- Test Set → final evaluation

This ensured realistic generalization measurement and cleaner experimentation.

---

# Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Future Improvements

Planned future experiments:
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting
- Cross-validation pipelines
- Ensemble approaches
- Spatial feature engineering

Tree-based and kernel-based models may better capture the nonlinear geographic relationships present in housing datasets.

---

# Repository Structure

```text
├── house-prices-linear_regression.ipynb
├── house-prices-polynomial_regression.ipynb
├── README.md
└── LICENSE
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
- nonlinear learning,
- optimization behavior,
- regularization,
- and practical regression modeling beyond tutorial-level implementation.

The repository is being continuously expanded with additional regression models and experimentation workflows.
