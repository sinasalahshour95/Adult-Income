# Comprehensive Adult Income Classification Pipeline

An end-to-end, production-grade machine learning project demonstrating a robust pipeline to predict whether an individual's income exceeds $50,000/year using US Census data. This repository features advanced workflows in data preprocessing, deep-learning-based and statistical data resampling, multiple feature selection paradigms, extensive hyperparameter optimization, and comparative model evaluation.

# Project Overview
Predicting income level from census metrics is a classic classification challenge often constrained by extreme class imbalance and high-dimensional categorical features. This project acts as a comparative framework demonstrating how different data balancing strategies (including standard resampling and a deep Autoencoder) combined with distinct feature selection filters/wrappers impact the predictive performance of traditional and ensemble machine learning algorithms. 

The pipeline is split modularly into two key notebooks:
1. **`Adult_Income.ipynb`**: Handles exploratory analysis, robust data cleaning, log transformations, categorical encoding, feature selection, and data resampling.
2. **`Training_AdultIncom.ipynb`**: Focuses on model selection, efficient hyperparameter tuning using successive halving, custom decision-boundary threshold tuning, and final metric comparisons.

# Pipeline Features

### 1. Data Preprocessing & Engineering
- **Missing Value Resolution:** Identifies and cleans implicit missing entries.
- **Feature Transformation:** Applies log transformations (`log_capital_gain`, `log_capital_loss`) to heavily skewed numeric financial distributions.
- **Categorical Encoding:** Leverages `OneHotEncoder` and intelligent binning/grouping to handle high-cardinality categorical attributes (such as `native_country`) into clean, binary flags.
- **Feature Scaling:** Employs `MinMaxScaler` across continuous dimensions to guarantee convergence for distance-based and neural network architectures.

### 2. Class Imbalance Resampling Methods
To address class imbalance (where `<=50K` dominates `>50K`), five alternative training configurations were generated and cross-compared:
- **Original Data:** The baseline dataset without modifications.
- **Random Over-Sampling (ROS)**.
- **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **SMOTETomek:** Combines SMOTE over-sampling with Tomek Links under-sampling to clear ambiguous noisy boundaries.
- **Deep Autoencoder (AE):** Trains a specialized Keras Autoencoder model to capture the reconstruction distribution of the data, utilizing it to generate unique synthesized minority class samples.

### 3. Feature Selection Strategies
- **Low-Variance Filter:** Eliminates constant or near-constant features.
- **Tree-Based Importance**
- **Mutual Information (MI):** Calculates information gain metrics between features and target.
- **Recursive Feature Elimination (RFE):** Employs wrapper-based recursive cross-validation (`RFECV`) to filter out less descriptive parameters.

### 4. Hyperparameter Tuning & Optimization
- **Successive Halving Search (`HalvingGridSearchCV`):** Used to search massive hyperparameter grids with an optimized allocation of resources.
- **Decision Threshold Tuning:** Custom functions evaluate classification probabilities to find the optimal decision boundary for optimizing the **F1-macro score**, rather than defaulting blindly to a 0.5 cutoff.

# Models Implemented
The project comprehensively trains, tunes, and evaluates a wide portfolio of estimators across every data configuration:
1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM)** with RBF Kernel
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier (GBC)**
5. **Balanced Random Forest Classifier** (specialized for imbalanced learning)
6. **XGBoost Classifier**
7. **Neural Network with RBF Activation** (Hyperparameter-tuned using Keras Tuner)

# Dataset
This project utilizes the **Adult Income Dataset**. 
