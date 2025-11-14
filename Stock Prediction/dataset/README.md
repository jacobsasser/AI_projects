# S&P 500 Technology Stocks + Macroeconomic Merge
This notebook cleans and merges three datasets:
1. S&P 500 companies list  
2. S&P 500 daily stock trading data  
3. A macroeconomic indicators dataset (interpolated to daily frequency)

The goal is to produce a unified dataset of technology-sector stock prices enriched with macroeconomic variables for downstream modeling.
This repository includes the fully cleaned and merged dataset:

`sp500_stocks_with_macro_clean.csv`

### **1. S&P 500 Daily Stock Market Data**  
**Source:** https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks  
Contains daily OHLCV data for all S&P 500 companies and company metadata.

### **2. Macroeconomic Indicators Dataset**  
**Source:** https://www.kaggle.com/datasets/sagarvarandekar/macroeconomic-factors-affecting-us-housing-prices  
Contains indicators such as unemployment rate, CPI, consumer confidence index, PPI, inflation, mortgage rates, GDP metrics, and more.
