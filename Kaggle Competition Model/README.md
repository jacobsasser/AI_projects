# Accident Risk Prediction (Kaggle Playground S5E10)

This project builds a deep learning model to predict **accident risk** using tabular road and environmental data. The approach combines **categorical embeddings**, a **PyTorch MLP**, and **Optuna hyperparameter tuning** to achieve strong predictive performance.

---

## Overview

- Task: Predict continuous accident risk scores  
- Dataset: Kaggle Playground Series S5E10  
- Features include road type, lighting, weather, traffic conditions, speed limits, etc.  
- Target: `accident_risk` (0–1)

---
## Dataset

This project uses the dataset from the Kaggle Playground Series S5E10 competition:

**Dataset:** https://www.kaggle.com/competitions/playground-series-s5e10

To run the notebook, download:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

and place them in the project directory.
## Approach

### **1. Data Preprocessing**
- Encode categorical features with integer codes  
- Scale numerical features via `StandardScaler`  
- Construct model-ready matrices for PyTorch  

### **2. Neural Model**
- Embedding layers for categorical columns  
- MLP for combined features (ReLU + BatchNorm + Dropout)  
- AdamW optimizer + CosineAnnealingLR  
- Trained with MSE loss and gradient clipping  

### **3. Hyperparameter Tuning**
- Automated search with Optuna (TPE sampler)  
- Tuned: learning rate, dropout, embedding dropout, batch size, weight decay, MLP architecture, early-stopping patience  
- 3-fold stratified CV (binning continuous target)  
- Exported trial results for analysis  

### **4. Final Training + Submission**
- Retrained best model on **100% of training data**  
- Generated predictions for test set  
- Saved Kaggle submission: `submission_mlp_fullfit.csv`

---

## Results

- Strong RMSE on validation folds  
- Competitive Kaggle leaderboard performance (Rank 757, RMSE result: 0.05578) 
- Optuna visualizations show LR, dropout, and MLP depth heavily influence performance

---

## Running the Project

```bash
pip install -r requirements.txt

Download the dataset from Kaggle, place train.csv and test.csv in the project directory, then run:

accidents.ipynb


The final submission file will be saved automatically.
