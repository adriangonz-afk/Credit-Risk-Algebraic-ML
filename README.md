# Algebraic Diagnosis of Credit Risk

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

---

## Why this project?

Many credit risk models fail not because of the algorithm, but because the feature space itself is unstable.

In real banking and financial environments, multicollinearity, redundant variables, and ill-conditioned matrices silently degrade model reliability, even when accuracy metrics appear acceptable.

This project focuses on diagnosing those algebraic issues before trusting any model.

---

## Project Overview

This repository analyzes the German Credit Risk dataset (UCI) from a Linear Algebra–first perspective, treating credit risk modeling as a matrix stability problem rather than a black-box classification task.

Instead of starting with Logistic Regression or Tree-based models, the project first asks:

> Is the system \( X\theta = y \) mathematically stable enough to trust its predictions?

---

## Mathematical Foundation

The credit risk problem is formulated as an overdetermined linear system:

\[
X\theta = y
\]

Two analytical approaches are contrasted.

---

### 1. Normal Equation (Classical OLS)

\[
\theta = (X^T X)^{-1} X^T y
\]

This approach becomes unreliable when:
- Features are highly correlated
- The Gram matrix \( X^T X \) is close to singular
- Small numerical noise is amplified into large parameter errors

---

### 2. Moore–Penrose Pseudoinverse (SVD)

To ensure numerical stability, the model uses Singular Value Decomposition:

\[
X = U \Sigma V^T
\quad \Rightarrow \quad
\theta = V \Sigma^+ U^T y
\]

This produces a minimum-norm solution that remains stable even when the system is ill-conditioned or redundant.

---

## Dataset

- Source: German Credit Data (UCI Machine Learning Repository)
- Observations: 1000 loan applicants
- Target Variable: Credit Default  
  - 0: Good Credit  
  - 1: Bad Credit
- Features:
  - Numerical: duration, credit amount, age
  - Categorical: employment, housing, loan purpose (One-Hot Encoded)

---

## Methodology

### 1. Feature Space Construction
- One-Hot Encoding of categorical variables
- Normalization of numerical features
- Explicit bias term inclusion

### 2. Algebraic Diagnosis
- Rank analysis to detect linear dependence
- Condition number to measure numerical instability
- SVD spectrum analysis to identify redundancy

### 3. Stability Testing
- Classical inversion using `numpy.linalg.inv`
- Robust solution using `numpy.linalg.pinv`

### 4. Financial Interpretation
- Mapping stable coefficients back to business logic
- Identifying variables that reduce or increase default risk

---

## Key Findings and Diagnosis

| Metric | Result | Interpretation |
|------|------|------|
| Rank | Full (49 / 49) | No exact multicollinearity |
| Condition Number | 7.34e+04 | Ill-conditioned system |
| SVD Spectrum | Steep decay | Strong dimensional redundancy |
| Model Accuracy | 78.60% | Strong linear signal |

A full-rank matrix does not guarantee numerical stability. High condition numbers can severely destabilize classical solutions.

---

## Final Algebraic Credit Risk Model

Based on SVD-derived coefficients, the final linear approximation of credit risk is:

\[
\hat{y} \approx 0.6090
- 0.2702(X_{A14})
- 0.2579(X_{A34})
- 0.2545(X_{A48})
- 0.2504(X_{A410})
\]

### Interpretation

- Base Risk (\( \beta_0 = 0.6090 \))  
  High baseline default probability across the population.

- Risk-Reducing Factors:
  - \( X_{A14} \): No checking account  
    Indicates absence of overdrafts or negative balances.
  - \( X_{A34} \): Critical or existing credit history  
    Demonstrates proven debt management.
  - \( X_{A48} \): Loan purpose related to education or retraining.
  - \( X_{A410} \): Business or other productive loan purposes.

These coefficients align with real-world financial intuition.

---

## Repository Structure

```text
credit-risk-algebraic-diagnosis/
├── data/
│   ├── raw/            # Original German Credit dataset
│   └── processed/      # Normalized matrices (X, y)
├── notebooks/
│   ├── 01_feature_engineering.ipynb
│   ├── 02_algebraic_diagnosis.ipynb
│   └── 03_model_stability.ipynb
└── README.md
