# Algebraic Diagnosis of Credit Risk

## Project Overview

This project approaches the Credit Risk classification problem through the lens of Linear Algebra rather than standard "black box" Machine Learning techniques. Using the German Credit Risk dataset, the primary objective is to diagnose the algebraic stability of the feature space before model training.

The core hypothesis is that blind application of logistic regression or decision trees on ill-conditioned data leads to unstable predictions. This project demonstrates how analyzing matrix properties—specifically Rank, Determinant, and Condition Number—can detect multicollinearity and structural redundancy that statistical summaries often miss.

## Mathematical Foundation

The project models credit risk as an overdetermined linear system:

$$X\theta = y$$

Where $X$ represents the client features matrix and $y$ the default risk vector. The analysis contrasts two approaches to finding the parameter vector $\theta$:

### 1. The Normal Equation (Classical OLS)
The standard analytical solution attempts to invert the Gram matrix ($X^T X$):

$$\theta = (X^T X)^{-1} X^T y$$

This method fails or produces numerically unstable results when columns in $X$ are linearly dependent (multicollinearity), causing the determinant to approach zero ($\det(X^T X) \approx 0$).

### 2. Moore-Penrose Pseudoinverse (SVD)
To resolve the singularity issue, this project implements the Singular Value Decomposition (SVD) method to compute the pseudoinverse ($X^+$). This creates a minimum-norm solution that remains stable even in the presence of redundant features:

$$\theta_{SVD} = V \Sigma^+ U^T y$$

## Dataset

* **Source**: German Credit Risk (UCI Machine Learning Repository).
* **Observations**: 1000 loan applicants.
* **Features**: A mix of numerical (Duration, Amount, Age) and categorical (Employment, Housing, Purpose) variables.

## Methodology

The repository is structured to simulate a rigorous engineering workflow:

1.  **Feature Space Construction**: Transformation of categorical variables via One-Hot Encoding and normalization of numerical vectors to ensure consistent basis alignment.
2.  **Algebraic Diagnosis**:
    * **Rank Analysis**: Comparing `Rank(X)` against the number of features to identify linear dependencies.
    * **Condition Number**: Calculating $\kappa(A)$ to quantify the system's sensitivity to noise.
3.  **Stability Testing**: Attempting to solve the system using both `numpy.linalg.inv` (standard inversion) and `numpy.linalg.pinv` (SVD-based pseudoinverse) to demonstrate the difference in numerical stability.
4.  **Financial Interpretation**: Mapping the resulting stable coefficients back to business logic to determine which variables truly drive credit default risk.

## Repository Structure

```text
credit-risk-algebraic-diagnosis/
├── data/
│   ├── raw/            # Original German Credit CSV
│   └── processed/      # Normalized matrices (X.npy, y.npy)
├── notebooks/
│   ├── 01_feature_engineering.ipynb    # Data cleaning and encoding
│   ├── 02_algebraic_diagnosis.ipynb    # Rank and Condition Number analysis
│   └── 03_model_stability.ipynb        # OLS vs SVD comparison
├── src/
│   ├── matrix_ops.py   # Helper functions for linear algebra
│   └── utils.py        # Data loading utilities
├── environment.yml     # Dependency definitions
└── README.md           # Project documentation
