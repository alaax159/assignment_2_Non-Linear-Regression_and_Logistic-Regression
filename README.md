# Assignment 2: Non-Linear Regression and Logistic Regression

This repository contains the implementation and analysis for **Assignment 2** of **ENCS5341 – Machine Learning and Data Science** at Birzeit University. The assignment explores two fundamental supervised learning techniques: non-linear regression using polynomial and radial basis function (RBF) features, and binary classification using logistic regression with feature engineering.

---

## Repository Structure
```
.
├── assignment_2.pdf                    # Original assignment description and requirements
├── Non-Linear-Regression.ipynb         # Notebook implementing polynomial and RBF regression
├── Logistic-Regression.ipynb           # Notebook implementing logistic regression for customer churn
├── customer_data.csv                   # Customer churn dataset
└── README.md                           # This file
```

**File Descriptions:**

- **`assignment_2.pdf`**: The official assignment document outlining the tasks, datasets, and evaluation criteria.
- **`Non-Linear-Regression.ipynb`**: Jupyter notebook containing experiments with polynomial regression (with ridge regularization) and RBF basis functions to model sinusoidal data with noise.
- **`Logistic-Regression.ipynb`**: Jupyter notebook that preprocesses customer churn data, trains multiple logistic regression models with varying feature complexity, and evaluates performance using standard classification metrics.
- **`customer_data.csv`**: Dataset containing customer information and churn labels used for the logistic regression task.

---

## Non-Linear Regression

### Overview

This section explores fitting a non-linear function using polynomial regression and radial basis functions (RBF). The goal is to understand the effects of model complexity and regularization on overfitting and generalization.

### Data Generation

- **Input space**: 100 points uniformly sampled from the interval [0, 1].
- **Target function**: y = sin(5πx) + ε, where ε represents Gaussian noise.
- The true underlying function is sinusoidal with 2.5 complete cycles in the unit interval.

### Polynomial Regression with Ridge Regularization

**Approach:**

- Fit polynomial regression models with ridge regularization (L2 penalty) to the generated data.
- Experiment with multiple regularization strengths (λ values), including λ = 0 (no regularization).
- Generate visualizations comparing:
  - Training data points
  - True underlying function (sin(5πx))
  - Fitted polynomial curves for each λ

**Key Findings:**

- **λ = 0 (No regularization)**: The model exhibits high variance and overfits the training noise, resulting in poor generalization.
- **Small λ**: Moderate regularization reduces overfitting while maintaining flexibility to capture the sinusoidal pattern.
- **Large λ**: Strong regularization leads to underfitting, as the model becomes too constrained and cannot capture the non-linear structure.
- The optimal λ balances bias and variance, achieving good fit on training data while generalizing well to unseen points.

### RBF Basis Functions

**Approach:**

- Replace polynomial features with Gaussian radial basis functions.
- Experiment with different numbers of basis functions: 1, 5, 10, and 50.
- Visualize the fitted curves and analyze model behavior.

**Key Findings:**

- **1 basis function**: Severe underfitting; the model lacks sufficient capacity to represent the sinusoidal pattern.
- **5 basis functions**: Captures the general shape but may miss finer details.
- **10 basis functions**: Good balance between fitting the data and avoiding overfitting.
- **50 basis functions**: Risk of overfitting; the model becomes too flexible and may fit noise rather than the underlying signal.
- RBF basis functions provide localized representations, making them effective for smooth function approximation when the number of basis functions is appropriately chosen.

---

## Logistic Regression

### Overview

This section applies logistic regression to predict customer churn using a real-world dataset. The task involves preprocessing, feature engineering, model training, and evaluation using multiple performance metrics.

### Dataset: Customer Churn

- **Source**: `customer_data.csv`
- **Description**: Contains customer demographic information, account details, and service usage patterns, along with a binary label indicating whether the customer churned.
- **Size**: Sufficient for train/validation/test splits (e.g., 2500/500/500 samples).

### Preprocessing Steps

1. **Handling Missing Values**: Imputation or removal of records with missing data.
2. **Encoding Categorical Features**: Conversion of categorical variables (e.g., gender, contract type) into numerical representations using one-hot encoding or label encoding.
3. **Feature Scaling**: Standardization or normalization of numerical features to ensure consistent scale across features (improves convergence and model performance).

### Model Training

**Feature Sets Tested:**

- **Linear features**: Original features without transformation.
- **Polynomial features**: Feature expansion using degrees 2, 5, and 9 to capture non-linear relationships.

**Training Strategy:**

- Split data into training (2500), validation (500), and test (500) sets.
- Train separate logistic regression models for each feature set.
- Tune hyperparameters and select the best model based on validation performance.

### Evaluation Metrics

Models are evaluated using:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of true positive predictions among all positive predictions (important for minimizing false alarms).
- **Recall**: Proportion of actual positives correctly identified (important for capturing all churners).

Performance is computed on training, validation, and test sets to assess overfitting and generalization.

### Model Selection

- The best model is selected based on validation set performance, balancing accuracy, precision, and recall.
- Typically, moderate polynomial degrees (e.g., degree 2) provide a good trade-off, improving over linear models without introducing excessive overfitting seen in high-degree polynomials (e.g., degree 9).

### ROC Curve and AUC

**Analysis:**

- The **Receiver Operating Characteristic (ROC) curve** is plotted for the selected model on the test set.
- The curve visualizes the trade-off between true positive rate (sensitivity) and false positive rate at various classification thresholds.
- **Area Under the Curve (AUC)**: A single scalar metric summarizing classifier performance.
  - AUC close to 1.0 indicates excellent discrimination between churners and non-churners.
  - AUC around 0.5 suggests random guessing.
- The test set AUC provides an unbiased estimate of the model's ability to generalize to new customers.

**Interpretation:**

The ROC curve and AUC help assess whether the model is suitable for deployment and guide threshold selection based on business requirements (e.g., prioritizing recall if retaining customers is critical).

---

## How to Run

### Requirements

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `seaborn` (optional, for enhanced visualizations)

### Installation

Install dependencies using pip:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn jupyter
```

Or using a requirements file (if provided):
```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. Clone or download this repository.
2. Navigate to the repository directory in your terminal.
3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open the desired notebook:
   - `Non-Linear-Regression.ipynb` for non-linear regression experiments.
   - `Logistic-Regression.ipynb` for logistic regression and churn prediction.

5. Run all cells sequentially to reproduce the analysis, visualizations, and results.

---

## Notes / Extensions

### Potential Improvements and Experiments

- **Non-Linear Regression:**
  - Experiment with different noise levels to study robustness.
  - Try alternative basis functions (e.g., sigmoid, wavelets).
  - Implement cross-validation for systematic λ selection.
  - Compare ridge regression with other regularization techniques (Lasso, Elastic Net).

- **Logistic Regression:**
  - Perform feature selection to identify the most predictive variables.
  - Experiment with class imbalance techniques (e.g., SMOTE, class weights) if churn is rare.
  - Try alternative classifiers (e.g., decision trees, random forests, gradient boosting) for comparison.
  - Conduct hyperparameter tuning using grid search or randomized search.
  - Analyze feature importance to gain business insights.

- **General:**
  - Add confidence intervals or bootstrapping for uncertainty quantification.
  - Create interactive visualizations using Plotly or Bokeh.
  - Document insights and recommendations in a final report.

---

## Contact

For questions or discussions related to this assignment, please contact the course instructor or refer to the course materials.

**Course**: ENCS5341 – Machine Learning and Data Science  
**Institution**: Birzeit University
