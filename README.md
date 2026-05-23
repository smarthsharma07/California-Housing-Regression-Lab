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
## 6. Support Vector Regression (SVR)

After experimenting with:
- Linear Regression
- Polynomial Regression
- Ridge/Lasso Regularization
- SGD-based optimization

the project moved toward kernel-based nonlinear learning using Support Vector Regression (SVR).

Unlike traditional regression models that attempt to fit a single global equation, SVR learns through:
- margin-based optimization
- similarity relationships
- nonlinear kernel transformations

This allowed the model to better capture:
- nonlinear geographic patterns
- clustered housing behavior
- local price relationships
- nonlinear income interactions

---

### Target Transformation

The target variable (`median_house_value`) showed:
- strong right skewness
- outlier-heavy behavior
- artificial clipping near \$500,000

To stabilize learning, a logarithmic transformation was applied:

```python
y_train_log = np.log1p(y_train)
```

Predictions were later restored using:

```python
np.expm1(predictions)
```

This significantly improved optimization stability and reduced the effect of extreme values.

---

## LinearSVR

The first experiment used `LinearSVR` as a baseline margin-based regression model.

Purpose:
- compare linear-margin learning against standard linear regression
- establish a kernel-free SVR baseline

Observations:
- captured some linear signal
- struggled heavily with nonlinear geographic behavior
- significantly underperformed compared to polynomial regression models

### Validation Performance

| Metric | Score |
|---|---|
| R² | ~0.22 |
| MAE | ~52.9k |
| RMSE | ~100.5k |

This demonstrated that:

> margin-based optimization alone is insufficient when the dataset contains strong nonlinear structure.

---

## RBF Kernel SVR

To model nonlinear relationships more effectively, the Radial Basis Function (RBF) kernel was introduced.

The RBF kernel works by measuring similarity between data points rather than fitting a single global linear equation.

This enabled the model to learn:
- local nonlinear relationships
- clustered spatial behavior
- nonlinear interactions automatically

without manually engineering higher-order polynomial features.

---

## Hyperparameter Tuning (GridSearchCV)

The RBF SVR model was tuned using `GridSearchCV`.

Parameters explored:

```python
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.1, 0.01],
    'epsilon': [0.1, 0.2, 0.5]
}
```

Cross-validation:
- 3-fold CV
- scoring metric: R²

### Best Parameters

```python
{
    'C': 10,
    'epsilon': 0.1,
    'gamma': 'scale'
}
```

### Validation Performance

| Metric | Score |
|---|---|
| R² | ~0.756 |
| MAE | ~35.7k |
| RMSE | ~56.3k |

This produced a massive improvement over:
- linear regression
- polynomial regression
- LinearSVR

---

## RandomizedSearchCV Tuned RBF SVR

To further optimize the model while reducing computational cost, `RandomizedSearchCV` was introduced.

Unlike exhaustive grid search, RandomizedSearchCV samples parameter combinations randomly, allowing:
- broader exploration
- faster experimentation
- lower computational overhead

while still achieving strong performance.

### Best Parameters

```python
{
    'gamma': 'scale',
    'epsilon': 0.01,
    'C': 5
}
```

---

## Final Validation Performance

| Metric | Value |
|---|---|
| MAE | ~35266.1 |
| RMSE | ~55285.21 |
| R² Score | ~0.7654 |


## Final Test Performance

| Metric | Value |
|---|---|
| MAE | ~34.3k |
| RMSE | ~53.3k |
| R² Score | ~0.785 |

This became the best-performing regression model in the project so far.

---

## Visualization Insights

An Actual vs Predicted scatter plot showed:
- strong alignment around the ideal prediction line
- accurate mid-range house predictions
- larger variance for expensive houses

The visualization also revealed prediction flattening near \$500k due to:
- target clipping within the California Housing dataset
- missing true luxury-house values

---

## Major SVR Insights

### Kernel Methods Capture Nonlinear Structure Effectively

The RBF kernel successfully modeled:
- nonlinear geographic relationships
- local housing clusters
- curved price behavior

far better than global linear models.

---

### Similarity-Based Learning Outperformed Explicit Polynomial Expansion

Polynomial Regression attempted to manually create nonlinear interactions.

RBF SVR instead learned nonlinear behavior through:

> similarity geometry between samples.

This produced significantly better generalization.

---

### Computational Cost Increased Dramatically

Kernel methods introduced:
- heavy CPU usage
- expensive cross-validation
- slow hyperparameter tuning

This highlighted an important real-world tradeoff between:
- predictive performance
- computational scalability

---

### Proper Experimental Discipline Was Critical

The workflow strictly maintained:
- Train Set → model fitting
- Validation Set → tuning
- Test Set → final evaluation

This ensured:
- minimal leakage
- realistic performance estimates
- reliable model comparison

---

# Updated Final Selected Model

Best Overall Model:
- RandomizedSearchCV Tuned RBF SVR
- RBF Kernel
- `C = 5`
- `epsilon = 0.01`
- `gamma = 'scale'`

---

---

# 7. Decision Tree Regression

After exploring:
- linear models
- polynomial feature expansion
- regularization techniques
- and kernel-based SVR models

the project moved toward tree-based nonlinear learning using Decision Tree Regression.

Unlike linear-family models that learn continuous mathematical equations, Decision Trees learn through:
- recursive feature splitting
- threshold-based partitioning
- piecewise prediction regions

This allows trees to naturally capture:
- nonlinear interactions
- feature thresholds
- clustered data behavior
- and local decision boundaries.

---

## Why Decision Trees?

The California Housing dataset contains:
- nonlinear geographic relationships
- clustered income behavior
- varying local housing patterns

Decision Trees are well-suited for:
- nonlinear tabular learning
- interpretable partition-based predictions
- automatic interaction discovery
- and feature-threshold learning.

---

# Preprocessing Strategy

Even though Decision Trees generally do not require feature scaling, the same preprocessing pipeline from previous experiments was reused for consistency and experimentation.

Applied preprocessing included:
- log transformations
- RobustScaler
- StandardScaler
- engineered ratio-based features

Experiments confirmed that scaling had minimal impact on standalone tree performance.

This reinforced an important ML concept:

> Tree-based models are largely scale-invariant because they split using thresholds rather than distance-based optimization.

---

# Simple Decision Tree Regressor

A baseline Decision Tree Regressor was first trained using manually selected regularization parameters.

## Model Configuration

```python
DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

---

## Validation Performance

| Metric | Value |
|---|---|
| MAE | ~40.4k |
| RMSE | ~60.1k |
| R² Score | ~0.723 |

---

## Observations

The Decision Tree:
- significantly outperformed Linear Regression
- performed better than Polynomial Regression models
- captured nonlinear structure effectively

However:
- predictions remained step-like
- the model showed instability compared to SVR
- smooth nonlinear approximation remained limited

This demonstrated one of the core characteristics of Decision Trees:

> Trees approximate nonlinear relationships using discrete partitions rather than continuous curves.

---

# Hyperparameter Tuning Using RandomizedSearchCV

To improve generalization and reduce overfitting, extensive hyperparameter tuning was performed using `RandomizedSearchCV`.

Unlike exhaustive grid search, randomized search:
- explores larger parameter spaces efficiently
- reduces computational cost
- allows broader experimentation

---

## Cross-Validation Strategy

Used:
- 5-Fold Cross Validation
- `scoring = "r2"`
- `n_iter = 80`

Importantly:
- tuning occurred only on the training set
- validation and test sets remained untouched

This ensured:
- minimal leakage
- reliable generalization measurement
- proper experimental discipline

---

## Parameters Explored

```python
param_dist = {
    "max_depth": [5, 10, 15, 20, 30, None],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 8, 10],
    "max_features": [None, "sqrt", "log2"],
    "criterion": ["squared_error", "friedman_mse"],
    "splitter": ["best", "random"],
    "ccp_alpha": [0.0, 0.0001, 0.001, 0.01, 0.1]
}
```

---

## Best Parameters Found

```python
{
    'min_samples_split': 5,
    'min_samples_leaf': 10,
    'max_features': None,
    'max_depth': 10,
    'criterion': 'squared_error'
}
```

---

## Validation Performance After Tuning

| Metric | Value |
|---|---|
| MAE | ~39k |
| RMSE | ~59k |
| R² Score | ~0.731 |

---

## Important Insight

Despite:
- larger search spaces
- pruning
- cross-validation
- regularization
- and aggressive tuning

performance improvements remained relatively small.

This highlighted an important machine learning concept:

> Sometimes model architecture becomes the bottleneck rather than hyperparameter tuning.

The experiments showed that standalone Decision Trees possess:
- limited smooth approximation ability
- high variance
- instability
- and diminishing returns from aggressive tuning.

---

# Final Test Set Evaluation

After selecting the best tuned Decision Tree model using validation performance, the model was evaluated on the completely unseen test dataset.

The test set remained untouched during:
- training
- tuning
- validation
- model selection

ensuring unbiased evaluation.

---

## Final Test Performance

| Metric | Value |
|---|---|
| MAE | ~37.5k |
| RMSE | ~55.9k |
| R² Score | ~0.763 |

---

## Generalization Analysis

Interestingly, the test performance slightly exceeded validation performance.

This suggested:
- strong generalization
- successful regularization
- limited overfitting
- and stable preprocessing behavior

The close agreement between validation and test metrics indicated that the tuning strategy generalized well beyond the validation split.

---

# Visualization Insights

When visualizing predictions against `median_income`, the Decision Tree produced:
- staircase-like prediction regions
- abrupt jumps
- flat prediction segments

instead of smooth regression curves.

This occurs because Decision Trees:
- partition feature space into discrete regions
- learn threshold-based rules
- make piecewise constant predictions

rather than learning continuous mathematical functions.

The visualization provided a strong intuition for how trees fundamentally differ from:
- linear regression
- polynomial regression
- and kernel-based SVR models.

---

# Major Decision Tree Insights

## Nonlinear Learning Without Explicit Feature Expansion

Unlike Polynomial Regression, Decision Trees automatically captured:
- nonlinear interactions
- threshold relationships
- and local feature behavior

without manually engineering higher-order polynomial terms.

---

## Tree Models Are Naturally Interpretable

Decision Trees provide:
- human-readable splits
- explicit decision boundaries
- transparent feature thresholding

making them highly interpretable compared to kernel methods.

---

## Single Trees Have Structural Limitations

Experiments showed that standalone trees:
- are unstable learners
- suffer from variance
- and plateau in performance relatively quickly

even after extensive hyperparameter tuning.

This naturally motivates:
- Random Forests
- Gradient Boosting
- XGBoost
- LightGBM

where multiple trees are combined to improve stability and predictive performance.

---

# Updated Model Ranking

Current performance hierarchy:

1. Tuned RBF SVR
2. Tuned Decision Tree Regressor
3. Polynomial + Ridge Regression
4. Polynomial + Lasso Regression
5. Polynomial Regression
6. SGDRegressor
7. Linear Regression
8. LinearSVR

---

# Final Test Results

| Metric | Value |
|---|---|
| MAE | ~34.3k |
| RMSE | ~53.3k |
| R² Score | ~0.785 |

The close agreement between validation and test metrics indicated:
- strong generalization
- stable preprocessing
- successful nonlinear learning
- effective hyperparameter tuning

---

# Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn


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

# Future Improvements

Planned future experiments:
- Random Forest Regressor
- Gradient Boosting
- XGBoost / LightGBM
- Cross-validation pipelines
- Ensemble approaches
- 
Tree-based and boosting models may further improve performance on nonlinear tabular housing datasets.

---

#  Repository Structure

```text
├── house-prices-linear_regression.ipynb
├── house-prices-polynomial_regression.ipynb
├── house-prices-svr-(rbf kernel).ipynb
├── house-prices-decisiontree.ipynb
├── README.md
└── LICENSE
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
