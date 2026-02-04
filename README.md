# Algebraic Diagnosis of Credit Risk

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![NumPy](https://img.shields.io/badge/Library-NumPy-orange) ![Status](https://img.shields.io/badge/Status-Completed-green)

## Project Overview

This project approaches the Credit Risk classification problem through the lens of **Linear Algebra** rather than standard "black box" Machine Learning techniques. Using the **German Credit Risk dataset** (UCI), the primary objective is to diagnose the algebraic stability of the feature space before model training.

The core hypothesis is that the blind application of logistic regression or decision trees on **ill-conditioned data** leads to unstable predictions. This project demonstrates how analyzing matrix properties—specifically Rank, Singular Value Decomposition (SVD), and Condition Number—can detect multicollinearity and structural redundancy that statistical summaries often miss.

---

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

---

## Dataset

* **Source**: [German Credit Data (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
* **Observations**: 1000 loan applicants.
* **Target**: 700 Good Credit (0), 300 Bad Credit (1).
* **Features**: A mix of numerical (Duration, Amount, Age) and categorical (Employment, Housing, Purpose) variables transformed via One-Hot Encoding.

---

## Methodology

The repository is structured to simulate a rigorous engineering workflow:

1.  **Feature Space Construction**: Transformation of categorical variables via One-Hot Encoding and normalization of numerical vectors to ensure consistent basis alignment.
2.  **Algebraic Diagnosis**:
    * **Rank Analysis**: Comparing `Rank(X)` against the number of features to identify linear dependencies.
    * **Condition Number**: Calculating $\kappa(A)$ to quantify the system's sensitivity to noise.
3.  **Stability Testing**: Attempting to solve the system using both `numpy.linalg.inv` (standard inversion) and `numpy.linalg.pinv` (SVD-based pseudoinverse) to demonstrate the difference in numerical stability.
4.  **Financial Interpretation**: Mapping the resulting stable coefficients back to business logic to determine which variables truly drive credit default risk.

---

## Key Findings & Diagnosis

The algebraic diagnosis of the matrix $X$ yielded critical insights:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Rank** | Full Rank (49/49) | No perfect multicollinearity exists. |
| **Condition Number** | **7.34e+04** | **Ill-Conditioned**. The system is highly sensitive to input noise. Standard inversion is risky. |
| **SVD Spectrum** | Steep Decay | Significant dimensional redundancy. A few features capture the majority of the variance. |
| **Model Accuracy** | **78.60%** | The linear signal is strong enough to predict default with high accuracy despite the condition number. |

---

## Final Algebraic Model

Based on the weights obtained via Singular Value Decomposition (SVD), we constructed a linear equation to approximate the **Credit Risk Score ($\hat{y}$)**. This equation quantifies how specific client characteristics increase or decrease the probability of default.

**Model Equation:**
$$\hat{y} \approx 0.6090 - 0.2702(X_{A14}) - 0.2579(X_{A34}) - 0.2545(X_{A48}) - 0.2504(X_{A410})$$

**Key Components:**
1.  **Base Risk ($\beta_0 = 0.6090$):** The intercept represents the starting probability of default (approx. 60%) for a client with no other distinguishing features. This indicates a high baseline risk in the population.
2.  **Risk Reducers (Negative Weights):**
    * **$X_{A14}$ (-0.2702):** Client has **no checking account**. This implies the absence of negative balances or overdrafts, acting as a strong stability indicator.
    * **$X_{A34}$ (-0.2579):** Client has **critical/existing credit history**. Successfully managing debts at other banks serves as proof of reliability.
    * **$X_{A48}$ (-0.2545):** Loan purpose is **Retraining/Education**.
    * **$X_{A410}$ (-0.2504):** Loan purpose is **Business/Other**.

---

## Repository Structure

```text
credit-risk-algebraic-diagnosis/
├── data/
│   ├── raw/            # Original German Credit CSV (german.data)
│   └── processed/      # Normalized matrices (X_matrix.npy, y_vector.npy)
├── notebooks/
│   ├── 01_feature_engineering.ipynb    # Data cleaning, One-Hot Encoding, and Matrix Generation
│   ├── 02_algebraic_diagnosis.ipynb    # Rank, SVD, and Condition Number analysis
│   └── 03_model_stability.ipynb        # OLS vs SVD comparison and Final Predictions
└── README.md           # Project documentation
```
## Getting Started
Clone the repository:

Bash
git clone [https://github.com/adriangonz-afk/Credit-Risk-Algebraic-ML.git](https://github.com/adriangonz-afk/Credit-Risk-Algebraic-ML.git)
cd Credit-Risk-Algebraic-ML
Run the notebooks:

    * Start with 01_feature_engineering.ipynb to download data from UCI and generate the matrices.

    * Proceed to 02_algebraic_diagnosis.ipynb to perform the health check on the matrix.

    * Finish with 03_model_stability.ipynb to see the final predictions and model weights.

###  Execute in Google Colab

| Notebook | Description | Link |
| :--- | :--- | :--- |
| **01. Feature Engineering** | Data ingestion, cleaning & matrix construction. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adriangonz-afk/Credit-Risk-Algebraic-ML/blob/main/notebooks/01_feature_engineering.ipynb) |
| **02. Algebraic Diagnosis** | Rank analysis & SVD stability check. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adriangonz-afk/Credit-Risk-Algebraic-ML/blob/main/notebooks/02_algebraic_diagnosis.ipynb) |
| **03. Model Stability** | OLS vs. SVD Comparison & Final Predictions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adriangonz-afk/Credit-Risk-Algebraic-ML/blob/main/notebooks/03_model_stability.ipynb) |
