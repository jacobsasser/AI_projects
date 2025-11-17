## Substance Abuse Treatment Completion Modeling (TEDS-D)

### Introduction

This project analyzes factors associated with **treatment relapse and non-completion** in substance abuse treatment programs across the **Northeastern United States**, using a subset of the national **TEDS-D (Treatment Episode Data Set – Discharge)** dataset. The goal is twofold:

1. **Identify key demographic, clinical, and treatment-related factors** that correlate with whether an individual completes treatment.
2. **Compare multiple machine learning models** to determine which approach most accurately predicts treatment completion, offering potential value for early intervention and resource allocation in treatment facilities.

The dataset includes variables such as substance use patterns, prior treatment history, mental health indicators, sources of referral, demographics, and treatment characteristics.

---

## Modeling: Predicting Treatment Completion

After the data was cleaned, geographically filtered, recoded from numeric codes into interpretable categories, and explored through correlation, histograms, and class distributions, the modeling phase focused on building predictive models for the binary outcome:  
**`completed_binary` = 1 if treatment was completed, 0 otherwise.**

### Modeling Pipeline

All models share a consistent preprocessing pipeline:

- **Split** into stratified train/test sets  
- **Numeric features** → median imputation + StandardScaler  
- **Categorical features** → most-frequent imputation + OneHotEncoder  
- Combined via **ColumnTransformer** and wrapped in scikit-learn **Pipelines** to ensure identical preprocessing across models.

### Models and Hyperparameter Tuning

Three supervised models were trained and tuned using GridSearchCV (3-fold CV, accuracy scoring):

---

#### **1. Random Forest Classifier**
- Tuned: `n_estimators` ∈ {50, 75, 100, 125, 150}  
- **Test Accuracy:** ~0.82  
- **Train Accuracy:** ~1.00 (indicating overfitting)  
- Visualizations: confusion matrix, feature importance, CV accuracy vs. n_estimators

---

#### **2. Logistic Regression**
- Tuned: `C` ∈ {0.001 → 100} (log scale)  
- **Test Accuracy:** ~0.77  
- Feature importance derived from absolute model coefficients  
- Most interpretable model in the study

---

#### **3. HistGradientBoostingClassifier**
- Tuned: `learning_rate` ∈ {0.001 → 0.1}  
- **Test Accuracy:** ~0.82  
- Comparable to Random Forest but with better generalization  
- Permutation importance used to identify the strongest predictors

---

### Results Summary

- **HistGradientBoosting** and **Random Forest** were the top performers, both reaching **~82% accuracy**, with balanced recall for completed vs. non-completed cases.  
- **Logistic Regression** performed slightly lower at **~77% accuracy**, but provided the clearest interpretability.  
- Random Forest exhibited **overfitting**, while HistGradientBoosting offered a more stable performance across train/test sets.  
- Feature importance analyses highlighted meaningful behavioral and demographic patterns in treatment success.

---

### Conclusion

The analysis indicates that **tree-based ensemble models**—particularly HistGradientBoosting—are the most effective approaches for predicting treatment completion within the Northeastern TEDS-D sample. These models capture complex interactions among substance use history, demographics, co-occurring issues, and treatment characteristics, offering actionable insights that could help treatment facilities identify high-risk individuals earlier.  

Logistic Regression remains valuable when interpretability is essential, but boosted and bagged tree models ultimately delivered the strongest predictive performance in this setting.
